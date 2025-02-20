


from pathlib import Path
import motor.motor_asyncio  
import pandas as pd
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from src.ElectricityBill.exception import CustomException
from src.ElectricityBill.logger import logger
from src.ElectricityBill.config_entity.config_params import DataIngestionConfig


# Load the environment variables
load_dotenv()


class MongoDBConnection:
    def __init__(self, uri, db_name, collection_name):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name


    async def __aenter__(self):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(self.uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        logger.info("Connected to MongoDB Database asynchronously")
        return self.collection
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
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

    async def import_data_from_mongodb(self):
        start_time = time.time()
        start_timestamp = datetime.now()
        try:
            logger.info("Starting data ingestion...")
            async with self.mongo_connection as collection:
                all_data = await self._fetch_all_data(collection)
                if all_data.empty:
                    logger.warning("No data found in MongoDB.")
                    return
                cleaned_data = self._clean_data(all_data)
                output_path = await self._save_data(cleaned_data)
                await self._save_metadata(start_time, start_timestamp, len(cleaned_data), output_path)
                logger.info("Data ingestion completed successfully.")
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise CustomException(e)

    async def _fetch_all_data(self, collection) -> pd.DataFrame:
        try:
            logger.info("Fetching data from MongoDB asynchronously...")
            batch_size = self.config.config_data.get('batch_size', 10000)
            combined_df = pd.DataFrame()
            cursor = collection.find({}, {'_id': 0})
            async for batch in cursor.batch_size(batch_size):
                batch_df = pd.DataFrame([batch])
                combined_df = pd.concat([combined_df, batch_df], ignore_index=True)
            return combined_df if not combined_df.empty else pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data asynchronously: {e}")
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

    async def _save_data(self, df: pd.DataFrame) -> Path:
        try:
            root_dir = self.config.config_data['root_dir']
            output_path = Path(root_dir) / "electricity_data.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"Data saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise CustomException(e)
    
    async def _save_metadata(self, start_time: float, start_timestamp: datetime, total_records: int, output_path: Path):
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



