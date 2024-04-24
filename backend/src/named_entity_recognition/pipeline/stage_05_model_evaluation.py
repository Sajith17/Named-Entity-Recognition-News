from named_entity_recognition.config.configuration import ConfigurationManager
from named_entity_recognition.components.model_evaluation import ModelEvaluation
from named_entity_recognition import logger

STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self): 
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.get_model()
        model_evaluation.evaluate()