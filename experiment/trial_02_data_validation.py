
import sys
sys.path.append('/home/western/ds_projects/electricity-bills')

import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from src.ElectricityBill.logger import logger
from src.ElectricityBill.constants import *
from src.ElectricityBill.utils.commons import *
from src.ElectricityBill.exception import CustomException
from ydata_profiling import ProfileReport


@dataclass
class DataValidationConfig:
    root_dir: str
    data_dir: str
    val_status: str
    all_schema: dict
    validated_data: str
    profile_report_name: str


class ConfigurationManager:
    def __init__(self, config_filepath: Path = DATA_VALIDATION_CONFIG_FILEPATH):
        try:
            self.config = read_yaml(config_filepath)
            create_directories([self.config['artifacts_root']])
        except Exception as e:
            logger.exception(f"Error initializing ConfigurationManager: {e}")
            raise CustomException(e, sys)

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            config = self.config['data_validation']
            create_directories([config['root_dir']])

            schema_path = Path(config['all_schema'])
            try:
                all_schema = read_yaml(schema_path)
            except Exception as e:
                logger.error(f"Error reading schema file: {e}")
                raise CustomException(f"Error reading schema file: {e}", sys)

            data_validation_config = DataValidationConfig(
                root_dir=config['root_dir'],
                data_dir=config['data_dir'],
                val_status=config['val_status'],
                all_schema=all_schema,
                validated_data=config['validated_data'],
                profile_report_name=config.get('profile_report_name', "data_profile.html"),
            )
            return data_validation_config
        except Exception as e:
            logger.exception(f"Error getting Data Validation config: {e}")
            raise CustomException(e, sys)


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.dtype_mapping = {
            "string": ["object", "category", "string"],
            "integer": ["int32", "int64"],
            "float": ["float32", "float64"],
            "boolean": ["bool"],
            "datetime": ["datetime64[ns]", "datetime64"]
        }

    def validate_all_columns(self) -> bool:
        try:
            overall_status = True
            validation_results = {}

            try:
                data = pd.read_parquet(self.config.data_dir)
            except Exception as e:
                logger.error(f"Error reading Parquet file: {e}")
                raise CustomException(f"Error reading Parquet file: {e}", sys)

            all_cols = list(data.columns)
            all_schema = self.config.all_schema

            for col in all_cols:
                schema = all_schema.get(col)
                if not schema:
                    validation_results[col] = "Column missing in schema"
                    overall_status = False
                    logger.error(f"Column {col} missing in schema")
                    continue

                actual_dtype = str(data[col].dtype)
                expected_dtype = schema['type']
                if actual_dtype not in self.dtype_mapping.get(expected_dtype, []):
                    validation_results[col] = f"Incorrect dtype: expected {expected_dtype}, got {actual_dtype}"
                    overall_status = False
                    logger.error(f"Column {col}: {validation_results[col]}")
                    continue

                constraints = schema.get('constraints', {})
                if 'enum' in constraints:
                    allowed_values = set(constraints['enum'])
                    invalid_values = set(data[col]) - allowed_values
                    if invalid_values:
                        validation_results[col] = f"Invalid categorical values: {invalid_values}"
                        overall_status = False
                        logger.error(f"Column {col}: {validation_results[col]}")
                        continue

                if 'min' in constraints and 'max' in constraints:
                    min_val, max_val = constraints['min'], constraints['max']
                    out_of_range = ~((data[col] >= min_val) & (data[col] <= max_val))
                    if out_of_range.any():
                        validation_results[col] = f"Out of range: [{min_val}, {max_val}]"
                        overall_status = False
                        logger.error(f"Column {col}: {validation_results[col]}")
                        continue

                validation_results[col] = "Valid"

            self._save_validation_results(validation_results)

            if overall_status:
                overall_status = self._save_validated_data(data)

            return overall_status

        except Exception as e:
            logger.exception(f"Error during validation: {e}")
            raise CustomException(e, sys)

    def _save_validation_results(self, validation_results: dict):
        val_status_path = self.config.val_status
        try:
            with open(val_status_path, 'w') as f:
                json.dump(validation_results, f, indent=4)
            logger.info(f"Validation results saved to {val_status_path}")
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")
            raise CustomException(f"Failed to save validation results: {e}", sys)

    def _save_validated_data(self, data: pd.DataFrame) -> bool:
        try:
            output_path = self.config.validated_data
            data.to_parquet(output_path, index=False)
            logger.info(f"Validated data saved to {output_path}")

            profile = ProfileReport(data, title="Validated Data Profile")
            report_path = Path(self.config.root_dir) / self.config.profile_report_name
            profile.to_file(report_path)
            logger.info(f"Profile report generated at: {report_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to save validated data or generate profile report: {e}")
            raise CustomException(f"Failed to save validated data or generate profile report: {e}", sys)


if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        data_validation_config = config_manager.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        validation_status = data_validation.validate_all_columns()

        if validation_status:
            print("Data validation successful!")
        else:
            print("Data validation failed.")

    except Exception as e:
        print(f"Error: {e}")



