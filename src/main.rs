use anyhow::{Error as E, Result};
use candle_core::{Device, IndexOp, Module as CandleModule, Tensor};
use candle_nn::{Dropout, Linear, VarBuilder, linear};
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use candle_transformers::models::mimi::candle_nn;
use hf_hub::api::sync::Api as HfApi;
use hf_hub::{Repo, RepoType};
use serde_json;
use tokenizers::Tokenizer;

struct SentenceTransformer {
    bert: BertModel,
    device: Device,
    tokenizer: Tokenizer,
}

impl SentenceTransformer {
    fn new(model_name: &str, device: Device) -> Result<Self> {
        let api = HfApi::new()?;
        let repo = api.repo(Repo::new(model_name.to_string(), RepoType::Model));
        let config_filename = repo.get("config.json")?;
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let weights_filename = repo.get("model.safetensors")?;

        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)?
        };
        let model = BertModel::load(vb, &config)?;

        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
            pp.pad_to_multiple_of = None;
        }

        if let Some(tp) = tokenizer.get_truncation_mut() {
            tp.max_length = 512;
            tp.strategy = tokenizers::TruncationStrategy::LongestFirst;
        }

        Ok(Self {
            tokenizer,
            bert: model,
            device,
        })
    }

    fn embed(&self, sentences: &[&str]) -> Result<Tensor> {
        let tokens = self
            .tokenizer
            .encode_batch(
                sentences.to_vec(),
                true, // add_special_tokens
            )
            .map_err(E::msg)?;

        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &self.device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;

        let attention_masks_2d_vec: Vec<Tensor> = tokens
            .iter()
            .map(|tokens| {
                // get_attention_mask() returns u32, which is what Tensor::new expects
                let mask = tokens.get_attention_mask().to_vec();
                Tensor::new(mask.as_slice(), &self.device).map_err(E::msg)
            })
            .collect::<Result<Vec<_>>>()?;
        let attention_mask_2d = Tensor::stack(&attention_masks_2d_vec, 0)?;

        // 2. The `token_type_ids` are all zeros for single sentences.
        let token_type_ids = token_ids.zeros_like()?;

        // 3. Get embeddings using the correct 2D attention mask.
        let embeddings =
            self.bert
                .forward(&token_ids, &token_type_ids, Some(&attention_mask_2d))?;
        let attention_mask_3d = attention_mask_2d.to_dtype(DTYPE)?.unsqueeze(2)?;

        // --- END: The Fix ---
        let cls_emb = embeddings.i((0, 0))?; // shape: [hidden]

        // print first few dims
        println!("CLS embedding shape: {:?}", cls_emb.dims());
        println!("CLS embedding first 5 dims: {:?}", cls_emb.narrow(0, 0, 5)?);
        // pooling
        let (b_size, n_tokens, _hidden) = embeddings.dims3()?;
        dbg!(b_size, n_tokens);

        let embeddings = (embeddings.broadcast_mul(&attention_mask_3d))?.sum(1)?;
        let sum_mask = attention_mask_3d.sum(1)?;
        let embeddings = embeddings.broadcast_div(&sum_mask)?;

        // normalization
        let embeddings = embeddings.broadcast_div(&embeddings.sqr()?.sum_keepdim(1)?.sqrt()?)?;

        Ok(embeddings)
    }
}

struct BertPooler {
    dense: Linear,
}

impl BertPooler {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        Ok(Self { dense })
    }
}

impl CandleModule for BertPooler {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        let xs = self.dense.forward(xs)?;
        xs.tanh()
    }
}
pub struct CrossEncoder {
    bert: BertModel,
    pooler: BertPooler,
    dropout: Dropout,
    classifier: Linear,
    tokenizer: Tokenizer,
    device: Device,
    max_length: usize,
}

impl CrossEncoder {
    pub fn new(model_name: &str, device: Device) -> Result<Self> {
        let api = HfApi::new().map_err(|e| candle_core::Error::Msg(format!("API error: {}", e)))?;
        let repo = api.model(model_name.to_string());

        let tokenizer_filename = repo
            .get("tokenizer.json")
            .map_err(|e| candle_core::Error::Msg(format!("Failed to download tokenizer: {}", e)))?;
        let config_filename = repo
            .get("config.json")
            .map_err(|e| candle_core::Error::Msg(format!("Failed to download config: {}", e)))?;
        let weights_filename = repo
            .get("model.safetensors")
            .map_err(|e| candle_core::Error::Msg(format!("Failed to download weights: {}", e)))?;

        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenizer error: {}", e)))?;
        let config_content = std::fs::read(&config_filename)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read config: {}", e)))?;
        let config: Config = serde_json::from_slice(&config_content)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to parse config: {}", e)))?;

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };

        let bert = BertModel::load(vb.pp("bert"), &config)?;
        let pooler = BertPooler::load(vb.pp("bert.pooler"), &config)?;
        let classifier = linear(config.hidden_size, 1, vb.pp("classifier"))?;

        let dropout = Dropout::new(config.hidden_dropout_prob as f32);

        let max_length = 512;

        Ok(Self {
            bert,
            pooler,
            dropout,
            classifier,
            tokenizer,
            device,
            max_length,
        })
    }

    pub fn predict(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>> {
        let mut scores = Vec::with_capacity(pairs.len());
        // TODO: batch
        for (query, passage) in pairs {
            let score = self.predict_pair(query, passage)?;
            scores.push(score);
        }
        Ok(scores)
    }

    fn predict_pair(&self, query: &str, passage: &str) -> Result<f32> {
        let encoding = self
            .tokenizer
            .encode((query, passage), true)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenization error: {}", e)))?;

        let tokens = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        let type_ids = encoding.get_type_ids();

        let max_len = std::cmp::min(tokens.len(), self.max_length);
        let tokens = &tokens[..max_len];
        let attention_mask = &attention_mask[..max_len];
        let type_ids = &type_ids[..max_len];

        let input_ids = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let attention_mask = Tensor::new(attention_mask, &self.device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::new(type_ids, &self.device)?.unsqueeze(0)?;

        let bert_output = self
            .bert
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        let cls_output = bert_output.i((.., 0))?; // Use '..' to keep the batch dim
        let pooled_output = self.pooler.forward(&cls_output)?;

        let dropped = self.dropout.forward(&pooled_output, false)?;

        let logits = self.classifier.forward(&dropped)?;

        let score = logits.squeeze(0)?.squeeze(0)?.to_scalar::<f32>()?;

        Ok(score)
    }

    pub fn rank(
        &self,
        query: &str,
        passages: &[&str],
        return_documents: bool,
    ) -> Result<Vec<RankResult>> {
        let pairs: Vec<(&str, &str)> = passages.iter().map(|p| (query, *p)).collect();
        let scores = self.predict(&pairs)?;

        let mut results: Vec<RankResult> = scores
            .into_iter()
            .enumerate()
            .map(|(idx, score)| RankResult {
                corpus_id: idx,
                score,
                text: if return_documents {
                    Some(passages[idx].to_string())
                } else {
                    None
                },
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        Ok(results)
    }
}

#[derive(Debug, Clone)]
pub struct RankResult {
    pub corpus_id: usize,
    pub score: f32,
    pub text: Option<String>,
}

fn main() -> Result<()> {
    println!("Loading cross-encoder model...");
    let model = CrossEncoder::new("cross-encoder/ms-marco-MiniLM-L6-v2", Device::Cpu)?;
    println!("Model loaded successfully!");

    let query = "How many people live in Berlin?";
    let passages = [
        "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
        "Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.",
        "In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.",
    ];

    println!("\n=== Predicting Scores ===");
    let pairs: Vec<(&str, &str)> = passages.iter().map(|p| (query, *p)).collect();
    let scores = model.predict(&pairs)?;

    println!("Scores: {:?}", scores);

    println!("\n=== Ranking Passages ===");
    let ranks = model.rank(query, &passages, true)?;

    println!("Query: {}", query);
    for rank in &ranks {
        println!(
            "- #{} ({:.2}): {}",
            rank.corpus_id,
            rank.score,
            rank.text.as_ref().unwrap()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_float_eq {
        ($left:expr, $right:expr, $eps:expr) => {{
            let (left, right, eps) = ($left, $right, $eps);
            let diff = (left - right).abs();
            if diff > eps {
                panic!(
                    r#"assertion failed: `(left ≈ right)`
  left: `{:?}`,
 right: `{:?}`,
  diff: `{:?}`,
   eps: `{:?}`"#,
                    left, right, diff, eps
                );
            }
        }};
    }

    fn cross_encoder(device: Device) -> Vec<f32> {
        let query = "How many people live in Berlin?";
        let passages = [
            "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
            "Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.",
            "In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.",
        ];
        let pairs: Vec<(&str, &str)> = passages.iter().map(|p| (query, *p)).collect();

        let model = CrossEncoder::new("cross-encoder/ms-marco-MiniLM-L6-v2", device).unwrap();
        let rust_answers = model.predict(&pairs).unwrap();

        rust_answers
    }

    fn cross_encoder_matches_python(device: Device, eps: f32) {
        let rust_answers = cross_encoder(device);

        let python_answers: [f32; _] = [8.607141, 5.5062656, 6.3529854];

        for (rust_answer, python_answer) in rust_answers.iter().zip(python_answers) {
            assert_float_eq!(rust_answer, python_answer, eps);
        }
    }

    fn cross_encoder_devices_agree(device1: Device, device2: Device, eps: f32) {
        let device1_answers = cross_encoder(device1);
        let device2_answers = cross_encoder(device2);

        for (device1_answer, device2_answer) in device1_answers.iter().zip(device2_answers) {
            assert_float_eq!(device1_answer, device2_answer, eps);
        }
    }

    fn metal() -> Device {
        Device::new_metal(0).unwrap()
    }

    #[test]
    fn test_cross_encoder_close_to_python_cpu() {
        cross_encoder_matches_python(Device::Cpu, 1e-6);
    }

    #[test]
    #[should_panic]
    fn test_cross_encoder_not_that_close_to_python_cpu() {
        cross_encoder_matches_python(Device::Cpu, 1e-7);
    }

    #[test]
    fn test_cross_encoder_close_to_python_metal() {
        cross_encoder_matches_python(metal(), 5e-6);
    }

    #[test]
    #[should_panic]
    fn test_cross_encoder_not_that_close_to_python_metal() {
        cross_encoder_matches_python(metal(), 1e-7);
    }

    #[test]
    fn test_cross_encoder_devices_close() {
        cross_encoder_devices_agree(Device::Cpu, metal(), 5e-6);
    }

    #[test]
    #[should_panic]
    fn test_cross_encoder_devices_not_that_close() {
        cross_encoder_devices_agree(Device::Cpu, metal(), 1e-7);
    }

    fn embeddings(device: Device) -> Vec<f32> {
        let sentences = [
            "The weather is lovely today.",
            "It's so sunny outside!",
            "He drove to the stadium.",
        ];

        let model =
            SentenceTransformer::new("sentence-transformers/all-MiniLM-L6-v2", device).unwrap();
        let embeddings: Tensor = model.embed(&sentences).unwrap();

        let similarities = embeddings.matmul(&embeddings.t().unwrap()).unwrap();
        let rust_answers: Vec<f32> = [(0, 1), (0, 2), (1, 2)]
            .iter()
            .map(|idx| similarities.i(*idx).unwrap().to_scalar::<f32>().unwrap())
            .collect();

        rust_answers
    }

    fn embeddings_matches_python(device: Device, eps: f32) {
        let rust_answers = embeddings(device);

        let python_answers: [f32; _] = [0.665955, 0.104584, 0.141145];

        for (rust_answer, python_answer) in rust_answers.iter().zip(python_answers) {
            assert_float_eq!(rust_answer, python_answer, eps);
        }
    }

    fn embeddings_devices_agree(device1: Device, device2: Device, eps: f32) {
        let device1_answers = embeddings(device1);
        let device2_answers = embeddings(device2);

        for (device1_answer, device2_answer) in device1_answers.iter().zip(device2_answers) {
            assert_float_eq!(device1_answer, device2_answer, eps);
        }
    }

    #[test]
    fn test_embeddings_close_to_python_cpu() {
        embeddings_matches_python(Device::Cpu, 1e-6);
    }

    #[test]
    #[should_panic]
    fn test_embeddings_not_that_close_to_python_cpu() {
        embeddings_matches_python(Device::Cpu, 1e-7);
    }

    #[test]
    fn test_embeddings_close_to_python_metal() {
        embeddings_matches_python(metal(), 5e-6);
    }

    #[test]
    #[should_panic]
    fn test_embeddings_not_that_close_to_python_metal() {
        embeddings_matches_python(metal(), 1e-7);
    }

    #[test]
    fn test_embeddings_devices_close() {
        embeddings_devices_agree(Device::Cpu, metal(), 5e-6);
    }

    #[test]
    #[should_panic]
    fn test_embeddings_devices_not_that_close() {
        embeddings_devices_agree(Device::Cpu, metal(), 1e-7);
    }
}
