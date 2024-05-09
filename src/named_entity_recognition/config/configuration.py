from named_entity_recognition.constants import *
from named_entity_recognition.utils.common import read_yaml, create_directories
from named_entity_recognition.entity.config_entity import (DataIngestionConfig,
                                                           EmbeddingPreparationConfig,
                                                           DataTransformationConfig,
                                                           ModelTrainerConfig,
                                                           ModelEvaluationConfig)
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
     
     def get_embedding_preparation_config(self) -> EmbeddingPreparationConfig:

        config = self.config.embedding_preparation

        create_directories([config.root_dir])

        embedding_preparation_config = EmbeddingPreparationConfig(
            root_dir = Path(config.root_dir),
            source_URL=config.source_URL,
            local_embedding_file=Path(config.local_embedding_file),
            unzip_dir=Path(config.unzip_dir),
            params_embedding_dims=self.params.EMBEDDING_DIM,
            params_vocab_size=self.params.VOCAB_SIZE
        )

        return embedding_preparation_config 
     
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
     
     def get_model_trainer_config(self) -> ModelTrainerConfig:

        config = self.config.model_trainer
        params = self.params
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir = Path(config.root_dir),
            data_path = Path(config.data_path),
            embedding_weights_path=Path(config.embedding_weights_path),
            params_num_encoder_layers=params.NUM_ENCODER_LAYERS,
            params_num_tags = params.NUM_TAGS,
            params_vocab_size = params.VOCAB_SIZE,
            params_embedding_dim = params.EMBEDDING_DIM,
            params_fully_connected_dim = params.FULLY_CONNECTED_DIM,
            params_num_heads = params.NUM_HEADS,
            params_max_positional_encoding_length = params.MAX_POSITIONAL_ENCODING_LENGTH,
            params_epochs = params.EPOCHS,
            params_batch_size = params.BATCH_SIZE
        )

        return model_trainer_config
     
     def get_model_evaluation_config(self):

        config = self.config.model_evaluation
        create_directories([config.root_dir])

        evaluation_config = ModelEvaluationConfig(
            root_dir = Path(config.root_dir),
            data_path = Path(config.data_path),
            model_path = Path(config.model_path),
            mlflow_uri = config.mlflow_uri,
            all_params = self.params,
            params_num_encoder_layers=self.params.NUM_ENCODER_LAYERS,
            params_num_tags = self.params.NUM_TAGS,
            params_vocab_size = self.params.VOCAB_SIZE,
            params_embedding_dim = self.params.EMBEDDING_DIM,
            params_fully_connected_dim = self.params.FULLY_CONNECTED_DIM,
            params_num_heads = self.params.NUM_HEADS,
            params_max_positional_encoding_length = self.params.MAX_POSITIONAL_ENCODING_LENGTH,
            params_batch_size = self.params.BATCH_SIZE,    
        )

        return evaluation_config

