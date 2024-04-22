from named_entity_recognition.config.configuration import ConfigurationManager
from named_entity_recognition.components.tokenizer_preparation import TokenizerPreparation
from named_entity_recognition import logger

STAGE_NAME = "Tokenizer Preparation stage"

class TokenizerPreparationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_tokenizer_preparation_config()
        data_ingestion = TokenizerPreparation(config=data_ingestion_config)
        data_ingestion.prepare_tokenizer()