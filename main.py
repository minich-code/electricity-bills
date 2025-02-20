
import sys
from src.ElectricityBill.logger import logger 
from src.ElectricityBill.pipelines.pip_01_data_ingestion import DataIngestionPipeline
from src.ElectricityBill.exception import CustomException


COMPONENT_01_NAME = "DATA INGESTION COMPONENT"
try:
    logger.info (f"------------> starting {COMPONENT_01_NAME} pipeline ------------->")
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.run()
    logger.info(f"------------> {COMPONENT_01_NAME} pipeline completed successfully ------------->")

except Exception as e:
    logger.error(f"Error in {COMPONENT_01_NAME} pipeline: {e}")
    raise CustomException(e, sys)

COMPONENT_02_NAME = "DATA VALIDATION COMPONENT"
