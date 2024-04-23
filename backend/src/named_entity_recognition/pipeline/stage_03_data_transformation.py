from named_entity_recognition.config.configuration import ConfigurationManager
from named_entity_recognition.components.data_transformation import DataTransformation
from named_entity_recognition import logger

STAGE_NAME = "Data Transformation stage"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(data_transformation_config)
        data_transformation.transform_data()

    

    
