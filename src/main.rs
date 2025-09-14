use anyhow::{Error as E, Result};
use candle_core::{Device, IndexOp, Tensor};
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
        // Device::new_metal(0)?
        Device::Cpu
    } else {
        Device::Cpu
    };

    println!("Using device {:?}", device);

    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        RepoType::Model,
        "ea78891063587eb050ed4166b20062eaf978037c".to_string(),
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
    println!("Token ids: {}", token_ids.to_string());

    // --- START: The Fix ---

    // 1. Create the 2D attention mask from the tokenizer output.
    let attention_masks_2d_vec: Vec<Tensor> = tokens
        .iter()
        .map(|tokens| {
            // get_attention_mask() returns u32, which is what Tensor::new expects
            let mask = tokens.get_attention_mask().to_vec();
            Tensor::new(mask.as_slice(), &device).map_err(E::msg)
        })
        .collect::<Result<Vec<_>>>()?;
    let attention_mask_2d = Tensor::stack(&attention_masks_2d_vec, 0)?;

    // 2. The `token_type_ids` are all zeros for single sentences.
    let token_type_ids = token_ids.zeros_like()?;

    // 3. Get embeddings using the correct 2D attention mask.
    let embeddings = model.forward(&token_ids, &token_type_ids, Some(&attention_mask_2d))?;
    println!("Embeddings shape: {:?}", embeddings.dims());

    // 4. For pooling, create a 3D version of the mask.
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

    // pairwise similarity
    let similarities = embeddings.matmul(&embeddings.t()?)?;

    println!("Pairwise similarities: \n{}", similarities.to_string());
    Ok(())
}
