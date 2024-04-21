import os
import datasets
import pickle
from named_entity_recognition import logger
from named_entity_recognition.utils.common import get_size
from named_entity_recognition.entity.config_entity import DataIngestionConfig

class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            with open(self.config.local_data_file, "wb") as f:
                data = datasets.load_dataset(self.config.source_name)
                pickle.dump(data,f)
            logger.info(f"{self.config.source_name} dataset downloaded successfully")

        else:
            logger.info(f"File already exists of size: {get_size(self.config.local_data_file)}")