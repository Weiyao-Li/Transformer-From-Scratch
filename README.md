# Transformer From Scratch

A compact, readable Transformer (encoder-decoder) implementation, organized by module.

## Modules
- `model.py` — Core Transformer components: `InputEmbedding`, `PositionalEncoding`, attention/FFN blocks, encoder/decoder stack, projection, and `build_transformer`.
- `dataset.py` — `BilingualDataset` that builds fixed-length inputs/labels with `[SOS]` / `[EOS]` / `[PAD]`, plus padding + causal masks.
- `train.py` — End-to-end pipeline: tokenizer creation/loading, `get_ds` data split/loaders, training loop with checkpointing/TensorBoard, greedy validation decoding.
- `config.py` — Central configuration (hyperparameters, language pair, paths, experiment name).

## Outputs (when training)
- `tokenizer_{lang}.json`
- `weights/tmodel_XX.pt`
- `runs/tmodel/`

## Experiment Results
- Dataset: `opus_books` (en → it)
- Setup: `d_model=512`, `seq_len=350`, `lr=1e-4`
- Colab script overrides: `batch_size=32`, `num_epochs=10`, `preload=None`, `model_folder=/content/drive/.../weights`, `tokenizer_file=/content/drive/.../vocab/tokenizer_{0}.json`
- Hardware: Google Colab (NVIDIA A100, CUDA)
- Metrics: CER / WER / BLEU (validation)

![loss curve](LOSS_CURVE_LINK)
