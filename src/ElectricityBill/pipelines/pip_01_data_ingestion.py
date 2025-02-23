

import sys
import time
from typing import Any
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

# Assuming the following paths and modules are correctly configured
sys.path.append('/home/western/ds_projects/electricity-bills')
from src.ElectricityBill.logger import logger
from src.ElectricityBill.exception import CustomException
from src.ElectricityBill.config_manager.config_settings import ConfigurationManager
from src.ElectricityBill.components.c_01_data_ingestion import DataIngestion

PIPELINE_NAME = "DATA INGESTION PIPELINE"


@dataclass
class PipelineData:
    """Represents the data passed between pipeline steps."""
    data_ingestion_config: Any
    ingested_data: pd.DataFrame = None  


class DataIngestionPipeline:
    """Orchestrates the data ingestion pipeline (simplified for data ingestion only)."""

    def __init__(self):
        self.config_manager = ConfigurationManager()

    def run(self):
        """Executes the data ingestion pipeline."""
        try:
            logger.info(f"## ================ Starting {PIPELINE_NAME} pipeline =======================")

            # Fetch configurations
            data_ingestion_config = self.config_manager.get_data_ingestion_config()

            # Ingest data
            ingested_data = self.ingest_data(data_ingestion_config)

            logger.info(f"## ================ {PIPELINE_NAME} pipeline completed successfully =======================")

        except CustomException as e:
            logger.error(f"Error during {PIPELINE_NAME} pipeline execution: {e}")
            raise 

    def ingest_data(self, config):
        """
        Ingests data, retrying on failure.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} - Fetching data from MongoDB...")

                data_ingestion = DataIngestion(config=config)
                data_ingestion.import_data_from_mongodb() 

                logger.info("Data ingestion completed successfully.")
                return 

            except Exception as e:
                logger.error(f"Data ingestion failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    logger.info("Retrying data ingestion...")
                else:
                    raise CustomException(f"Data ingestion failed after {max_retries} attempts: {e}", sys)
        return None 


if __name__ == "__main__":
    try:
        # Instantiate and run the pipeline
        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.run()

    except Exception as e:
        logger.error(f"Error in {PIPELINE_NAME}: {e}")
        sys.exit(1)