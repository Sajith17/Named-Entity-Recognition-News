from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_name: str
    local_data_file: Path

@dataclass(frozen=True)
class TokenizerPreparationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_path: Path
