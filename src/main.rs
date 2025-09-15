use candle_core::{Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Dropout, Linear, VarBuilder, linear};
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::api::sync::Api as HfApi;
use tokenizers::Tokenizer;

pub struct BertPooler {
    dense: Linear,
}

impl BertPooler {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        Ok(Self { dense })
    }
}

impl Module for BertPooler {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
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
    pub fn new(model_name: &str) -> Result<Self> {
        let device = Device::Cpu;

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
    let model = CrossEncoder::new("cross-encoder/ms-marco-MiniLM-L6-v2")?;
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

    fn cross_encoder(eps: f32) {
        let query = "How many people live in Berlin?";
        let passages = [
            "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
            "Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.",
            "In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.",
        ];
        let pairs: Vec<(&str, &str)> = passages.iter().map(|p| (query, *p)).collect();

        let model = CrossEncoder::new("cross-encoder/ms-marco-MiniLM-L6-v2").unwrap();
        let scores = model.predict(&pairs).unwrap();

        let python_answer: [f32; _] = [8.607141, 5.5062656, 6.3529854];

        for (rust_answer, python_answer) in scores.iter().zip(python_answer) {
            assert_float_eq!(rust_answer, python_answer, eps);
        }
    }

    #[test]
    #[should_panic]
    fn test_cross_encoder_not_that_close() {
        cross_encoder(1e-9);
    }

    #[test]
    fn test_cross_encoder_close() {
        cross_encoder(1e-6);
    }
}
