
from pathlib import Path
import os
import sys
from dotenv import load_dotenv
from src.ElectricityBill.exception import CustomException
from src.ElectricityBill.logger import logger
from src.ElectricityBill.constants import *
from src.ElectricityBill.utils.commons import read_yaml, create_directories
from src.ElectricityBill.config_entity.config_params import DataIngestionConfig

# Load the environment variables
load_dotenv()


class ConfigurationManager:
    def __init__(self, 
                 data_ingestion_config: Path = DATA_INGESTION_CONFIG_FILEPATH,
                 data_validation_config: Path = DATA_VALIDATION_CONFIG_FILEPATH,
                 schema_config: Path = SCHEMA_CONFIG_FILEPATH,

    
    ):
        
        try:
            self.ingestion_config = read_yaml(data_ingestion_config)
            self.data_val_config = read_yaml(data_validation_config)
            self.schema = read_yaml(schema_config) 


            create_directories([self.ingestion_config['artifacts_root']])
            create_directories([self.data_val_config.artifacts_root])



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

# Data Validation 

    def get_data_validation_config(self) -> Configuration:
        try:
            data_valid_config = self.data_val_config.data_validation 
            schema_dict = self._process_schema()
            profile_report_path = os.path.join(data_valid_config.root_dir, "data_profile_report.html") # Define profile report path
            create_directories([Path(data_valid_config.root_dir)]) 
            logger.info(f"Data Validation Config Loaded") 

            return Configuration(DataValidationConfig(
                root_dir = Path(data_valid_config.root_dir), 
                val_status = data_valid_config.val_status, 
                data_dir = Path(data_valid_config.data_dir), 
                all_schema = schema_dict,
                critical_columns = data_valid_config.critical_columns,
                profile_report_path=profile_report_path  #Store path to config
            ))
        except Exception as e: 
            logger.exception(f"Error getting data validation configuration: {str(e)}") 
            raise CustomException(e, sys)

    def _process_schema(self) -> Dict[str, str]:
        schema_columns = self.schema.get("columns", {})
        target_column = self.schema.get("target_column", [])
        schema_dict = {col['name']: col['type'] for col in schema_columns}
        schema_dict.update({col['name']: col['type'] for col in target_column})
        return schema_dict
    
    

