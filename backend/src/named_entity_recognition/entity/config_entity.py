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
