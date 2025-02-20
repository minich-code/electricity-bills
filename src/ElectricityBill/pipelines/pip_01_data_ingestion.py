
import sys 
sys.path.append('/home/western/DS_Projects/electricity-bills')

import asyncio  
from src.ElectricityBill.exception import CustomException
from src.ElectricityBill.logger import logger
from src.ElectricityBill.config_manager.config_settings import ConfigurationManager
from src.ElectricityBill.components.c_01_data_ingestion import DataIngestion

PIPELINE_NAME = "DATA INGESTION PIPELINE"

class DataIngestionPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()
            user_name = config_manager.get_user_name()
            data_ingestion = DataIngestion(config=data_ingestion_config, user_name=user_name)
            asyncio.run(data_ingestion.import_data_from_mongodb())  # Updated to run async function
            logger.info("Data ingestion process completed successfully.")
        except CustomException as e:
            logger.error(f"Error during data ingestion: {e}")
            logger.info("Data ingestion process failed.")


if __name__ == "__main__":
    try:
        logger.info (f"------------> starting {PIPELINE_NAME} pipeline ------------->")
        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.run()
        logger.info(f"------------> {PIPELINE_NAME} pipeline completed successfully ------------->")

    except Exception as e:
        logger.error(f"Error in {PIPELINE_NAME} pipeline: {e}")
        raise CustomException(e, sys)
