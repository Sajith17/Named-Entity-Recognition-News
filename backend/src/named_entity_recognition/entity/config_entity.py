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
