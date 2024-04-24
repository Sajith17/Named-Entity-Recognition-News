from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str

@dataclass(frozen=True)
class TokenizerPreparationConfig:
    root_dir: Path
    data_path: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    dataset_name: str
    data_path: Path
    tokenizer_path: Path
    params_max_sequence_length: int
    params_label_all_tokens: bool

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    params_num_encoder_layers: int
    params_num_tags: int
    params_vocab_size: int
    params_embedding_dim: int
    params_fully_connected_dim: int
    params_num_heads: int
    params_max_positional_encoding_length: int
    params_epochs: int
    params_batch_size: int

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_weights_path: Path
    params_num_encoder_layers: int
    params_num_tags: int
    params_vocab_size: int
    params_embedding_dim: int
    params_fully_connected_dim: int
    params_num_heads: int
    params_max_positional_encoding_length: int
    params_batch_size: int
