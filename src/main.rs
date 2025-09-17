use anyhow::{Error as E, Result};
use candle_core::{Device, IndexOp, Module as CandleModule, Tensor};
use candle_nn::{Dropout, Linear, VarBuilder, linear};
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use candle_transformers::models::mimi::candle_nn;
use clap::{Parser, Subcommand};
use hf_hub::api::sync::Api as HfApi;
use hf_hub::{Repo, RepoType};
use indicatif::{ProgressBar, ProgressStyle};
use jiff::Timestamp;
use rusqlite::{Connection as SqlConnection, Name, Result as SqlResult, params};
use safetensors;
use serde_json;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tokenizers::Tokenizer;
use tracing::{info, info_span, instrument};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};
use tracing_timing;
use twox_hash::XxHash3_64;
use walkdir::{DirEntry, WalkDir};

static SKIP_DIRS: OnceLock<HashSet<&'static str>> = OnceLock::new();

fn is_hidden(s: &str) -> bool {
    s.starts_with(".") && s != "." && s != ".."
}

fn should_skip_dir(dir: &DirEntry) -> bool {
    let skip_dirs = SKIP_DIRS.get_or_init(|| {
        let dirs = [
            ".git",
            ".venv",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
            "target",
            "build",
            "dist",
        ];
        let answer = HashSet::from(dirs);
        answer
    });

    dir.file_name()
        .to_str()
        .map(|s| is_hidden(s) || skip_dirs.contains(s))
        .unwrap_or(false)
}

static SKIP_EXTS: OnceLock<HashSet<&'static str>> = OnceLock::new();

fn should_skip_ext(ext: &str) -> bool {
    let skip_exts = SKIP_EXTS.get_or_init(|| {
        let exts = [
            "pyc", "pyo", "webp", "so", "dylib", "dll", "exe", "bin", "jpg", "jpeg", "png", "gif",
            "h5", "pdf", "zip", "tar", "gz",
        ];
        let answer = HashSet::from(exts);
        answer
    });

    skip_exts.contains(ext)
}

fn should_skip_file(path: &Path) -> bool {
    // don't check whether path is a file to avoid a syscall. I'm okay accepting
    // incorrect behavior in weird cases where a directory is named "foo.png"
    // and we skip it in order to do better on the common case where there are
    // directories with thousands of .png files.
    return path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(should_skip_ext)
        .unwrap_or(false);
}

fn should_skip(entry: &DirEntry) -> bool {
    should_skip_dir(entry) || should_skip_file(entry.path())
}

struct HashOfStr(u64);

struct FileInfo {
    id: FileId,
    path: PathBuf,
    mtime: Timestamp,
    hash: HashOfStr,
}

#[derive(Copy, Clone)]
struct FileId(u64);

#[derive(Clone, Copy, Debug)]
struct ChunkRange {
    start_byte: u64,
    end_byte: u64,
}
struct Chunk {
    file: FileId,
    text: String,
    range: ChunkRange,
    hash: HashOfStr,
}

fn hash_content(content: &str) -> HashOfStr {
    HashOfStr(XxHash3_64::oneshot(content.as_bytes()))
}

// TODO opt file cache should maybe mmap files rather than read them
// TODO opt cpu embedder vs gpu embedder. or one for querying one for indexing
struct AppState {
    embedder: SentenceTransformer,
    cross_encoder: CrossEncoder,
    conn: SqlConnection,
    // embeddings: Tensor
    // file_cache: maybe Str -> mmap'd file descriptors at some point?
    // for now str -> str map
}

impl AppState {
    #[instrument]
    fn new() -> Result<Self> {
        // This is cheap and correct to clone because the GPU buffers and
        // command queues are in Arcs.
        let metal = Device::new_metal(0)?;
        let embedder =
            SentenceTransformer::new("sentence-transformers/all-MiniLM-L6-v2", metal.clone())?;
        let cross_encoder = CrossEncoder::new("cross-encoder/ms-marco-MiniLM-L6-v2", metal)?;
        let conn = SqlConnection::open("index.db")?;

        Ok(Self {
            embedder,
            cross_encoder,
            conn,
        })
    }

    fn chunk_text(content: &str, max_tokens: usize) -> Vec<ChunkRange> {
        // TODO opt make this more precise. Probably what we can do is tokenize
        // the whole thing using the embedding tokenizer, then take [max_tokens]
        // at a time, and work backwards to figure out the text ranges. idk if
        // the tokenizer supports that.

        // TODO cor later error if truncation happened

        // Rough estimate: 4 chars = 1 token
        let max_chars = (max_tokens * 4 * 7) / 10;

        let mut answer = Vec::new();
        let mut this_chunk_start = 0;

        let content = content.as_bytes();

        while this_chunk_start < content.len() {
            let mut this_chunk_end = this_chunk_start + max_chars;
            if this_chunk_end >= content.len() {
                answer.push(ChunkRange {
                    start_byte: this_chunk_start as u64,
                    end_byte: content.len() as u64,
                });
                break;
            }

            // Find last word boundary

            // TODO cor this doesn't handle UTF-8 properly
            while this_chunk_end > this_chunk_start
                && !content[this_chunk_end as usize].is_ascii_whitespace()
            {
                this_chunk_end -= 1;
            }

            // If no space found, just cut at max_chars
            if this_chunk_end == this_chunk_start {
                this_chunk_end = this_chunk_start + max_chars;
            }

            answer.push(ChunkRange {
                start_byte: this_chunk_start as u64,
                end_byte: this_chunk_end as u64,
            });

            this_chunk_start = this_chunk_end;

            // Skip whitespace
            while this_chunk_start < content.len()
                && content[this_chunk_start].is_ascii_whitespace()
            {
                this_chunk_start += 1;
            }
        }

        tracing::info!("{:?}", answer);
        answer
    }

    fn process_file(
        entry: DirEntry,
        max_tokens: usize,
        file_id: FileId,
    ) -> Result<(FileInfo, Vec<Chunk>)> {
        let file_mtime = entry.metadata().map_err(E::msg)?.modified()?;
        let file_mtime = jiff::Timestamp::try_from(file_mtime)?;
        // TODO cor deal with non-UTF binary files
        let content = std::fs::read_to_string(entry.path())?;
        let file_hash = hash_content(&content);
        let file_info = FileInfo {
            id: file_id,
            path: entry.into_path(),
            mtime: file_mtime,
            hash: file_hash,
        };

        let chunk_ranges = Self::chunk_text(&content, max_tokens);

        // TODO qual to more cool stuff here
        let filewide_tags = if let Some(filename) = file_info.path.file_name() {
            format!("Tags: {}. Content: ", filename.display())
        } else {
            "".to_string()
        };

        let mut answer = Vec::new();

        let content = content.as_bytes();
        for (chunk_idx, chunk_range) in chunk_ranges.iter().enumerate() {
            let original_text =
                &content[chunk_range.start_byte as usize..chunk_range.end_byte as usize];
            let embed_text = filewide_tags.clone() + &String::from_utf8_lossy(original_text);

            let hash = hash_content(&embed_text);

            let chunk = Chunk {
                file: file_id,
                text: embed_text,
                range: *chunk_range,
                hash,
            };

            answer.push(chunk);
        }

        Ok((file_info, answer))
    }

    #[instrument(skip_all)]
    fn write_embeddings(embeddings: Tensor, model_name: &str, path: &Path) -> Result<()> {
        let mut metadata = HashMap::new();
        metadata.insert("model_name".to_string(), model_name.to_string());
        // TODO put other metadata like creation time and sg version number
        let mut tensors = HashMap::new();
        tensors.insert("embeddings".to_string(), embeddings);
        safetensors::serialize_to_file(tensors, &Some(metadata), path)?;

        Ok(())
    }

    #[instrument(skip_all)]
    fn create_index(&mut self, directory: &str, batch_size: usize) -> Result<()> {
        // TODO opt tokenize metadata/tags just once
        // TODO opt tokenize the whole document, then create chunks of exactly the
        // right size
        let max_tokens = self.embedder.max_length();

        let mut all_files: Vec<FileInfo> = Vec::new();
        let mut all_chunks: Vec<Chunk> = Vec::new();

        for (file_idx, entry) in WalkDir::new(directory)
            .follow_links(true)
            .into_iter()
            .filter_entry(|e| !should_skip(e))
            .enumerate()
        {
            let entry = match entry {
                Ok(entry) => entry,
                Err(e) => {
                    tracing::error!("Skipping error dir {:?}", e);
                    continue;
                }
            };

            if entry.path().is_dir() {
                continue;
            }

            let filename_for_print = entry.clone();
            let (file_info, mut chunks) =
                Self::process_file(entry, max_tokens, FileId(file_idx as u64))?;
            tracing::info!(
                "File {} with {} chunks",
                filename_for_print.file_name().display(),
                chunks.len()
            );
            all_files.push(file_info);
            all_chunks.append(&mut chunks);
        }

        let embeddings = {
            let _span = tracing::info_span!(
                "embedding batches",
                total_files = all_files.len(),
                total_chunks = all_chunks.len(),
                batch_size = batch_size
            );

            let mut embeddings: Vec<Tensor> = Vec::new();

            let pb = ProgressBar::new(all_chunks.len() as u64);
            pb.set_style(
    ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} batches ({eta})")
        .unwrap()
        .progress_chars("#>-")
);
            pb.set_position(0);
            for (batch_idx, batch) in all_chunks.chunks(batch_size).enumerate() {
                pb.set_message(format!("Batch {} chunks", batch.len()));
                let chunk_texts: Vec<&str> = batch.iter().map(|x| x.text.as_str()).collect();
                let batch_embeddings = self.embedder.embed(&chunk_texts)?;
                embeddings.push(batch_embeddings);
                pb.set_position(((batch_idx + 1) * batch_size) as u64);
            }

            pb.finish_with_message("Embedding complete!");
            Tensor::cat(&embeddings, 0)?
        };

        Self::write_embeddings(
            embeddings,
            self.embedder.model_name(),
            Path::new("embeddings.safetensors"),
        )?;

        // TODO corr add a keywords table and keyword_id in chunks
        self.conn.execute_batch(
            "
            CREATE TABLE files (
                id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                mtime REAL NOT NULL,
                hash TEXT NOT NULL
            );

            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL,
                start_byte INTEGER NOT NULL,
                end_byte INTEGER NOT NULL,
                chunk_hash TEXT NOT NULL,
                matrix_index INTEGER NOT NULL,
                FOREIGN KEY (file_id) REFERENCES files (id)
            );

            CREATE INDEX idx_chunks_matrix_index ON chunks(matrix_index);
        ",
        )?;

        Ok(())
    }
}

#[derive(Parser, Debug)]
struct Args {
    /// The directory to index
    dir: String,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::{prelude::*, registry::Registry};

    let args = Args::parse();

    let (chrome_layer, _guard) = ChromeLayerBuilder::new().build();
    tracing_subscriber::registry()
        .with(chrome_layer)
        .with(tracing_subscriber::fmt::layer())
        .with(EnvFilter::from_default_env())
        // .with(
        //     tracing_timing::Builder::default()
        //         .layer(|| tracing_timing::Histogram::new_with_max(1_000_000, 2).unwrap()),
        // )
        .init();

    let mut app = AppState::new()?;
    app.create_index(args.dir.as_str(), 128)?;

    Ok(())
}

struct SentenceTransformer {
    bert: BertModel,
    device: Device,
    tokenizer: Tokenizer,
    model_name: String,
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
            model_name: model_name.to_string(),
        })
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn max_length(&self) -> usize {
        self.tokenizer.get_truncation().unwrap().max_length
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
        // let cls_emb = embeddings.i((0, 0))?; // shape: [hidden]

        // print first few dims
        // println!("CLS embedding shape: {:?}", cls_emb.dims());
        // println!("CLS embedding first 5 dims: {:?}", cls_emb.narrow(0, 0, 5)?);
        // pooling
        // let (b_size, n_tokens, _hidden) = embeddings.dims3()?;
        // dbg!(b_size, n_tokens);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_skip_ext() {
        assert_eq!(should_skip_ext("pyc"), true);
    }

    #[test]
    fn test_should_skip_file() {
        assert_eq!(should_skip_file(Path::new("foo.txt")), false);
        assert_eq!(should_skip_file(Path::new("foo.pyc")), true);
        assert_eq!(should_skip_file(Path::new("foo.exe")), true);
        assert_eq!(should_skip_file(Path::new("foo.")), false);
        assert_eq!(should_skip_file(Path::new("foo")), false);
        assert_eq!(should_skip_file(Path::new(".foo")), false);
        assert_eq!(should_skip_file(Path::new(".foo.txt")), false);
        assert_eq!(should_skip_file(Path::new(".foo.pyc")), true);
        assert_eq!(should_skip_file(Path::new("foo.tar.gz")), true);
        assert_eq!(should_skip_file(Path::new("foo.megalong")), false);
    }

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
