from named_entity_recognition.config.configuration import ConfigurationManager
from named_entity_recognition.components.model_trainer import ModelTrainer
from named_entity_recognition import logger

STAGE_NAME = "Model Training stage"

class ModelTrainingPipeline:
    def __init__(self): 
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.get_model()
        model_trainer.train()
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

