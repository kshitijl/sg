use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::{
    bert::{BertModel, Config, DTYPE},
    mimi::candle_nn,
};
use hf_hub::{Repo, RepoType, api::sync::Api};
use serde_json;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else if candle_core::utils::metal_is_available() {
        Device::new_metal(0)?
    } else {
        Device::Cpu
    };

    println!("Using device {:?}", device);

    let api = Api::new()?;
    let repo = api.repo(Repo::new(
        "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        RepoType::Model,
    ));
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

    let sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ];

    // tokenize

    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
        pp.pad_to_multiple_of = None;
    }

    if let Some(tp) = tokenizer.get_truncation_mut() {
        tp.max_length = 512;
        tp.strategy = tokenizers::TruncationStrategy::LongestFirst;
    }
    let tokens = tokenizer
        .encode_batch(
            sentences.to_vec(),
            true, // add_special_tokens
        )
        .map_err(E::msg)?;

    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), &device)?)
        })
        .collect::<Result<Vec<_>>>()?;

    let token_ids = Tensor::stack(&token_ids, 0)?;
    dbg!(token_ids.to_string());

    // get embeddings
    let embeddings = model.forward(&token_ids, &token_ids.zeros_like()?, None)?;
    println!("Embeddings shape: {:?}", embeddings.dims());

    // pooling
    let (b_size, n_tokens, _) = embeddings.dims3()?;
    dbg!(b_size, n_tokens);

    let attention_mask = token_ids.ne(0i64)?.to_dtype(DTYPE)?.unsqueeze(2)?;
    // let attention_mask = token_ids.ne(0f32)?.unsqueeze(2)?.to_dtype(DTYPE)?;
    let embeddings = (embeddings.broadcast_mul(&attention_mask))?.sum(1)?;
    let sum_mask = attention_mask.sum(1)?;
    let embeddings = embeddings.broadcast_div(&sum_mask)?;

    // normalization
    let embeddings = embeddings.broadcast_div(&embeddings.sqr()?.sum_keepdim(1)?.sqrt()?)?;

    // pairwise similarity
    let similarities = embeddings.matmul(&embeddings.t()?)?;

    println!("Pairwise similarities: \n{}", similarities.to_string());
    Ok(())
}
