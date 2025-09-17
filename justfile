check:
    cargo check --color=always 2>&1 | less -R

run:
    RUST_BACKTRACE=full RUST_LOG=info cargo run --release -- ~/notes

debug:
    RUST_BACKTRACE=full RUST_LOG=info cargo run -- ~/notes

clean:
    rm -f trace*.json index.db embeddings.safetensors
