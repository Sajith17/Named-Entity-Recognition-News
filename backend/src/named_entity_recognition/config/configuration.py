from named_entity_recognition.constants import *
from named_entity_recognition.utils.common import read_yaml, create_directories
from named_entity_recognition.entity.config_entity import (DataIngestionConfig,
                                                           TokenizerPreparationConfig,
                                                           DataTransformationConfig)
from pathlib import Path

class ConfigurationManager:
     def __init__(
               self, 
               config_filepath = CONFIG_FILE_PATH,
               params_filepath = PARAMS_FILE_PATH):
               self.config = read_yaml(config_filepath)
               self.params = read_yaml(params_filepath)

               create_directories([self.config.artifacts_root])
    
     def get_data_ingestion_config(self) -> DataIngestionConfig:
          config = self.config.data_ingestion

          create_directories([config.root_dir])

          data_ingestion_config = DataIngestionConfig(
               root_dir=Path(config.root_dir),
               dataset_name=config.dataset_name
          )
         
          return data_ingestion_config
     
     def get_tokenizer_preparation_config(self) -> TokenizerPreparationConfig:

        config = self.config.tokenizer_preparation

        create_directories([config.root_dir])

        tokenizer_preparation_config = TokenizerPreparationConfig(
            root_dir = Path(config.root_dir),
            data_path = Path(config.data_path)
        )

        return tokenizer_preparation_config 
     
     def get_data_transformation_config(self) -> DataTransformationConfig:

        config = self.config.data_transformation
        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir = Path(config.root_dir),
            dataset_name = self.config.data_ingestion.dataset_name,
            data_path = Path(config.data_path),
            tokenizer_path = Path(config.tokenizer_path),
            params_max_sequence_length=self.params.MAX_SEQUENCE_LENGTH,
            params_label_all_tokens=self.params.LABEL_ALL_TOKENS
        )
        return data_transformation_config