import os
from pathlib import Path
from datasets import load_dataset
from named_entity_recognition import logger
from named_entity_recognition.utils.common import get_size
from named_entity_recognition.entity.config_entity import DataIngestionConfig

class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        local_data_file = os.path.join(self.config.root_dir,self.config.dataset_name)
        if not os.path.exists(local_data_file):
            data = load_dataset(self.config.dataset_name)
            data.save_to_disk(local_data_file)
            logger.info(f"{self.config.dataset_name} dataset downloaded successfully")

        else:
            logger.info(f"File already exists of size: {get_size(local_data_file)}")