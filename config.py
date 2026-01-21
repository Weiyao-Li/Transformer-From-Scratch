from pathlib import Path


def get_config():
    return {
        "batch_size": 8,  # Number of samples per training batch
        "num_epochs": 20,  # Total number of training epochs
        "lr": 10 ** -4,  # Learning rate
        "seq_len": 350,  # Fixed sequence length (after padding)
        "d_model": 512,  # Transformer embedding size
        "lang_src": "en",  # Source language (encoder side)
        "lang_tgt": "it",  # Target language (decoder side)
        "model_folder": "weights",  # Folder to save model checkpoints
        "model_basename": "tmodel_",  # Base name for saved model files
        "preload": None,  # Path to pretrained weights (None = train from scratch)
        "tokenizer_file": "tokenizer_{0}.json",  # Tokenizer file template
        "experiment_name": "runs/tmodel"  # TensorBoard experiment name
    }


def get_weights_file_path(config, epoch: str):
    # Folder where model checkpoints are stored
    model_folder = config['model_folder']

    # Base name of the model file (e.g., "tmodel_")
    model_basename = config['model_basename']

    # Final filename: tmodel_1.pt, tmodel_2.pt, ...
    model_filename = f"{model_basename}{epoch}.pt"

    # Full path to the weight file
    return str(Path('.') / model_folder / model_filename)
