from named_entity_recognition.config.configuration import ConfigurationManager
from named_entity_recognition.components.embedding_preparation import EmbeddingPreparation
from named_entity_recognition import logger

STAGE_NAME = "Embedding Preparation stage"

class EmbeddingPreparationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        embedding_preparation_config = config.get_embedding_preparation_config()
        embedding_preparation = EmbeddingPreparation(embedding_preparation_config)
        embedding_preparation.download_file()
        embedding_preparation.unzip_file()
        embedding_preparation.prepare_word_to_vec()
        embedding_preparation.prepare_tokenizer()
        embedding_preparation.prepare_embedding_weights()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EmbeddingPreparationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e