
import sys
sys.path.append('/home/western/ds_projects/electricity-bills')

from dataclasses import dataclass
from pathlib import Path
import pymongo  # Use the synchronous pymongo driver
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from src.ElectricityBill.exception import CustomException
from src.ElectricityBill.logger import logger
from src.ElectricityBill.constants import DATA_INGESTION_CONFIG_FILEPATH
from src.ElectricityBill.utils.commons import read_yaml, create_directories

# Load the environment variables
load_dotenv()

@dataclass
class DataIngestionConfig:
    """
    A dataclass to hold data ingestion configuration.
    """
    config_data: dict


class ConfigurationManager:
    def __init__(self, data_ingestion_config: Path = DATA_INGESTION_CONFIG_FILEPATH):
        try:
            self.ingestion_config = read_yaml(data_ingestion_config)
            create_directories([self.ingestion_config['artifacts_root']])
            logger.info("Configuration directories created successfully.")

        except Exception as e:
            logger.error(f"Error initializing ConfigurationManager: {e}")
            raise CustomException(e, sys)
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            data_config = self.ingestion_config['data_ingestion']
            create_directories([data_config['root_dir']])
            logger.info(f"Data ingestion configuration loaded from: {DATA_INGESTION_CONFIG_FILEPATH}")
            data_config['mongo_uri'] = os.environ.get('MONGO_URI')
            return DataIngestionConfig(config_data=data_config)
        except Exception as e:
            logger.error(f"Error loading data ingestion configuration: {e}")
            raise CustomException(e, sys)
    
    def get_user_name(self):
        try:
            return self.ingestion_config['data_ingestion'].get('get_user_name', 'DefaultUser')
        except Exception as e:
            logger.error(f"Error getting user name from config: {e}")
            raise CustomException(e, sys)

class MongoDBConnection:
    def __init__(self, uri, db_name, collection_name):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None  # Initialize client to None
        self.db = None
        self.collection = None

    def __enter__(self):
        self.client = pymongo.MongoClient(self.uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        logger.info("Connected to MongoDB Database synchronously")
        return self.collection

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:  # Check if client exists before closing
            self.client.close()
            logger.info("MongoDB connection closed.")

class DataIngestion:
    def __init__(self, config: DataIngestionConfig, user_name: str):
        self.config = config
        self.user_name = user_name
        self.mongo_connection = MongoDBConnection(
            config.config_data['mongo_uri'],
            config.config_data['database_name'],
            config.config_data['collection_name']
        )

    def import_data_from_mongodb(self):
        start_time = time.time()
        start_timestamp = datetime.now()
        try:
            logger.info("Starting data ingestion...")
            with self.mongo_connection as collection:
                all_data = self._fetch_all_data(collection)
                if all_data.empty:
                    logger.warning("No data found in MongoDB.")
                    return
                cleaned_data = self._clean_data(all_data)
                output_path = self._save_data(cleaned_data)
                self._save_metadata(start_time, start_timestamp, len(cleaned_data), output_path)
                logger.info("Data ingestion completed successfully.")
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise CustomException(e)


    def _fetch_all_data(self, collection) -> pd.DataFrame:
        try:
            logger.info("Fetching data from MongoDB synchronously...")
            #batch_size = self.config.config_data.get('batch_size', 10000) #no longer used
            combined_df = pd.DataFrame()
            cursor = collection.find({}, {'_id': 0})
            
            # Iterate through all documents in the cursor
            for batch in cursor:
                batch_df = pd.DataFrame([batch])  # Wrap the single document in a list
                combined_df = pd.concat([combined_df, batch_df], ignore_index=True)
            
            return combined_df if not combined_df.empty else pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data synchronously: {e}")
            raise CustomException(e)
        
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Drop columns with all unique values (including index-like columns)
            unique_cols = [col for col in df.columns if df[col].nunique() == len(df)]
            df = df.drop(columns=unique_cols)
            
            # Drop zero variance columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            zero_var_cols = [col for col in numeric_cols if df[col].var() == 0]
            df = df.drop(columns=zero_var_cols)
            
            logger.info(f"Dropped unique value columns: {unique_cols}")
            logger.info(f"Dropped zero variance columns: {zero_var_cols}")
            
            return df
        except Exception as e:
            logger.error(f"Error during data cleaning: {e}")
            raise CustomException(e)

    def _save_data(self, df: pd.DataFrame) -> Path:
        try:
            root_dir = self.config.config_data['root_dir']
            output_path = Path(root_dir) / "electricity_data.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"Data saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise CustomException(e)
    
    def _save_metadata(self, start_time: float, start_timestamp: datetime, total_records: int, output_path: Path):
        try:
            root_dir = self.config.config_data['root_dir']
            metadata = {
                'start_time': start_timestamp.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': time.time() - start_time,
                "total_records": total_records,
                "data_source": self.config.config_data['collection_name'],
                "ingested_by": self.user_name,
                "output_path": str(output_path)
            }
            metadata_path = Path(root_dir) / "data-ingestion-metadata.json"

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            logger.info("Metadata saved successfully.")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            raise CustomException(e)


if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        user_name = config_manager.get_user_name()
        data_ingestion = DataIngestion(config=data_ingestion_config, user_name=user_name)
        data_ingestion.import_data_from_mongodb()
        logger.info("Data ingestion process completed successfully.")
    except CustomException as e:
        logger.error(f"Error during data ingestion: {e}")
        logger.info("Data ingestion process failed.")