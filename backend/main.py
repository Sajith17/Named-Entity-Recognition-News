from named_entity_recognition import logger
from named_entity_recognition.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline, STAGE_NAME


try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e



