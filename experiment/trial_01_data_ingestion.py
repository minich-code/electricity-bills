
import sys
sys.path.append('/home/western/ds_projects/electricity-bills')

from dataclasses import dataclass
from pathlib import Path
from pymongo  import MongoClient
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
load_dotenv()

@dataclass
class DataIngestionConfig:
    """
    A dataclass to hold data ingestion configuration.
    """
    root_dir: str
    database_name: str
    collection_name: str
    batch_size: int
    mongo_uri: str


class ConfigurationManager:
    def __init__(self, data_ingestion_config: str = DATA_INGESTION_CONFIG_FILEPATH):  
        try:
            logger.info(f"Initializing ConfigurationManager with config file: {data_ingestion_config}")
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
            mongo_uri = os.environ.get('MONGO_URI')
            return DataIngestionConfig(
                root_dir=data_config['root_dir'],
                database_name=data_config['database_name'],
                collection_name=data_config['collection_name'],
                batch_size=int(data_config['batch_size']), 
                mongo_uri=mongo_uri)
        except Exception as e:
            logger.error(f"Error loading data ingestion configuration: {e}")
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
        self.client = MongoClient(self.uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        logger.info("Connected to MongoDB Database synchronously")
        return self.collection

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:  
            self.client.close()
            logger.info("MongoDB connection closed.")

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.mongo_connection = MongoDBConnection(
            self.config.mongo_uri,  
            self.config.database_name, 
            self.config.collection_name 
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
            raise CustomException(e, sys) # Ensure sys is passed to CustomException


    def _fetch_all_data(self, collection) -> pd.DataFrame:
        try:
            logger.info("Fetching data from MongoDB...")
            batch_size = self.config.batch_size
            data_list = []

            # Use cursor with batch_size for efficient memory usage
            cursor = collection.find({}, {'_id': 0}).batch_size(batch_size)
            for document in cursor:
                data_list.append(document)

                # Optional: Process in batches to avoid memory issues
                if len(data_list) >= batch_size:
                    df_batch = pd.DataFrame(data_list)
                    if 'combined_df' not in locals():
                        combined_df = df_batch
                    else:
                        combined_df = pd.concat([combined_df, df_batch], ignore_index=True)
                    data_list = []

            # Process any remaining documents
            if data_list:
                df_batch = pd.DataFrame(data_list)
                if 'combined_df' not in locals():
                    combined_df = df_batch
                else:
                    combined_df = pd.concat([combined_df, df_batch], ignore_index=True)

            return combined_df if 'combined_df' in locals() else pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise CustomException(e, sys)
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the DataFrame by dropping columns with zero variance and unique values.
        Replicates the simpler notebook logic.
        """
        try:
            # Identify columns with zero variance (nunique == 1)
            zero_variance_columns = [col for col in df.columns if df[col].nunique() == 1]

            # Drop columns with zero variance
            df.drop(columns=zero_variance_columns, errors='ignore', inplace=True)
            logger.info(f"Removed columns with zero variance: {zero_variance_columns}")

            # Identify columns with unique values
            unique_value_columns = [col for col in df.columns if df[col].nunique() == len(df)]

            # Drop columns with unique values
            df.drop(columns=unique_value_columns, errors='ignore', inplace=True)
            logger.info(f"Removed columns with unique values: {unique_value_columns}")

            # Replace infinite values with NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Drop any rows that contain NaN values
            df.dropna(inplace=True)

            logger.info("Data cleaning completed successfully.")
            return df

        except Exception as e:
            logger.error(f"Error during data cleaning: {e}")
            raise CustomException(e, sys)

    
    def _save_data(self, df: pd.DataFrame) -> Path:
        try:
            root_dir = self.config.root_dir
            output_path = Path(root_dir) / "electricity_data.parquet"  # Corrected filename
            df.to_parquet(output_path, index=False)
            logger.info(f"Data saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise CustomException(e, sys) 

    def _save_metadata(self, start_time: float, start_timestamp: datetime, total_records: int, output_path: Path):
        try:
            root_dir = self.config.root_dir
            metadata = {
                'start_time': start_timestamp.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': time.time() - start_time,
                "total_records": total_records,
                "data_source": self.config.collection_name,
                "output_path": str(output_path)
            }
            metadata_path = Path(root_dir) / "data-ingestion-metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            logger.info("Metadata saved successfully.")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            raise CustomException(e, sys)  # Ensure sys is passed to CustomException


if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.import_data_from_mongodb()
        logger.info("Data ingestion process completed successfully.")
    except CustomException as e:
        logger.error(f"Error during data ingestion: {e}")
        logger.info("Data ingestion process failed.")

