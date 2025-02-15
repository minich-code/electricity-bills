import sys
sys.path.append('/home/western/DS_Projects/electricity-bills')


from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, List, Any
from collections import defaultdict
import pandas as pd
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import json
import sys
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from src.ElectricityBill.logger import logger
from src.ElectricityBill.constants import *
from src.ElectricityBill.utils.commons import *
from src.ElectricityBill.exception  import CustomException
from ydata_profiling import ProfileReport  

@dataclass
class ValidationResult:
    """Encapsulates the result of a validation check."""
    status: bool
    errors: Optional[Dict[str, Any]] = None

    def __bool__(self):
        return self.status
    
    def __str__(self):
        return f"Status: {self.status}, Errors: {self.errors}"

class Configuration:
    def __init__(self, config: 'DataValidationConfig'):
          self._config = config

    def get_config(self) -> 'DataValidationConfig':
        return self._config
    
    def set_config(self, new_config: 'DataValidationConfig') -> None:
        self._config = new_config
    
    def __str__(self):
         return f"Configuration:\n{self._config}"

@dataclass
class DataValidationConfig:
    """
    Configuration for data validation process.

    """
    root_dir: Path
    val_status: str
    data_dir: Path
    all_schema: Dict[str, Any]
    critical_columns: List[str]
    profile_report_path: str  # Add path for the profile report

class ConfigurationManager:
    def __init__(
              self,
              data_validation_config: Path = DATA_VALIDATION_CONFIG_FILEPATH,
              schema_config: Path = SCHEMA_CONFIG_FILEPATH) -> None:
        
        logger.info(f"Initializing the configuration manager") 

        try:
            self.data_val_config = read_yaml(data_validation_config)
            self.schema = read_yaml(schema_config) 
            create_directories([self.data_val_config.artifacts_root]) 
        except Exception as e:
            logger.exception(f"Error creating directories") 
            raise CustomException(e, sys) 

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

class BaseValidator(ABC):
      def __init__(self, config: Configuration):
            self.config = config
            self.logger = logging.getLogger(self.__class__.__name__) 
      
      @abstractmethod
      def validate(self, data: pd.DataFrame, profile_report: Optional[Dict] = None) -> ValidationResult: #Profile Report passed to validation
            pass

class ColumnValidator(BaseValidator):
    """Validates the presence of required columns and absence of extra columns."""
    
    def validate(self, data: pd.DataFrame, profile_report: Optional[Dict] = None) -> ValidationResult:
        try:
            all_cols = frozenset(data.columns)
            all_schema = frozenset(self.config.get_config().all_schema.keys())

            missing_columns = list(all_schema - all_cols)
            extra_columns = list(all_cols - all_schema)

            if missing_columns or extra_columns:
                error_message = {
                    'missing_columns': missing_columns,
                    'extra_columns': extra_columns
                }
                self.logger.error(f"Column validation failed: {error_message}")
                return ValidationResult(status=False, errors=error_message)

            self.logger.info("Column validation passed")
            return ValidationResult(status=True)

        except Exception as e:
            self.logger.error(f"An error occurred during column validation: {e}")
            return ValidationResult(status=False, errors={"error": str(e)})


class DataTypeValidator(BaseValidator):
    """Validates that columns have the correct data types."""
    
    def validate(self, data: pd.DataFrame, profile_report: Optional[Dict] = None) -> ValidationResult:
        dtype_mapping = {
            "string": ["object", "category", "string"],
            "integer": ["int32", "int64"],
            "float": ["float32", "float64"],
            "boolean": ["bool"],
            "datetime": ["datetime64[ns]", "datetime64"]
        }

        all_schema = self.config.get_config().all_schema
        type_mismatches = {}
        validation_status = True
        string_value_errors = {}

        for col, expected_type in all_schema.items():
            if col not in data.columns:
                continue

            actual_type = str(data[col].dtype)
            if isinstance(expected_type, dict):
                expected_type = expected_type.get("type")

            if not isinstance(expected_type, str):
                self.logger.debug(f"Invalid data type for column {col}: {expected_type}")
                continue

            expected_pandas_type = dtype_mapping.get(expected_type)
            if expected_pandas_type and actual_type not in expected_pandas_type:
                type_mismatches[col] = {"expected": expected_type, "actual": actual_type}
                validation_status = False
            
            # string value check 
            if expected_type == "string" and col in data.columns:
                col_schema = next(
                    (schema for schema in self.config.get_config().all_schema.get("columns", []) if schema.get("name") == col),
                    None
                )
                if col_schema:
                    string_errors = self._validate_string_column(data, col, col_schema)
                    if string_errors:
                        string_value_errors.update(string_errors)
                        validation_status = False
        
        if string_value_errors:
            type_mismatches.update(string_value_errors)


        if not validation_status:
            self.logger.error(f"Data type validation failed: {type_mismatches}")
            return ValidationResult(status=False, errors = type_mismatches)
        else:
            self.logger.info("Data type validation passed")
            return ValidationResult(status=True)

    def _validate_string_column(self, data: pd.DataFrame, col: str, col_schema: Dict) -> Optional[Dict[str, List[str]]]:
        """Helper method to check string values against enums"""
        invalid_values = {}
        constraints = col_schema.get("constraints", [])

        def extract_enum_values(constraints: List[Dict[str, Any]]) -> Optional[List[str]]:
            for constraint in constraints:
                if constraint["type"] == "enum":
                    return constraint["values"]
            return None

        enum_values = extract_enum_values(constraints)
        if enum_values:
            enum_set = set(enum_values)
            if not data[col].isin(enum_set).all():
               invalid_values[col] = data[~data[col].isin(enum_set)][col].tolist()
        
        return invalid_values if invalid_values else None


class MissingValueValidator(BaseValidator):
    """Validates that critical columns do not have any missing values."""
    
    def validate(self, data: pd.DataFrame, profile_report: Optional[Dict] = None) -> ValidationResult:
        if not isinstance(data, pd.DataFrame):
            self.logger.error("Input data is not a DataFrame")
            return ValidationResult(status=False, errors = {"error": "Input data is not a DataFrame"})

        # Use profile_report to get missing value counts for critical columns
        missing_values = {}
        if profile_report:
            for col in self.config.get_config().critical_columns:
                if col in profile_report.variables: #Ensure column appears in report
                    missing_count = profile_report.variables[col].get("missing", 0)
                    if missing_count > 0:
                         missing_values[col] = missing_count
                else:
                   missing_values[col] = "Column not in profile report"  # Flag if column not present

        if missing_values:
            self.logger.debug(f"Missing value check failed: {missing_values}")
            return ValidationResult(status=False, errors = missing_values)
        self.logger.info(f"Missing value check passed") 
        return ValidationResult(status=True)

class ConstraintValidator(BaseValidator):
     """Validate that numeric-type columns contain only allowed values or fall within the specified range"""

     def validate(self, data: pd.DataFrame, profile_report: Optional[Dict] = None) -> ValidationResult:
          all_schema = self.config.get_config().all_schema.get("columns", {})
          constraint_violations = defaultdict(lambda: {"enum_violation": [], "range_violation": []})

          for col, col_schema in all_schema.items():
                col_type = col_schema.get("type", None)
                constraints = col_schema.get("constraints", [])

                if col_type in ["integer", "float"] and col in data.columns:
                    enum_values = next((c["values"] for c in constraints if c["type"] == "enum"), None)
                    if enum_values:
                        enum_set = set(enum_values)
                        if not data[col].isin(enum_set).all():
                            constraint_violations[col]["enum_violation"] = data[~data[col].isin(enum_set)][col].tolist()
                  
                    range_violations = self._check_range_violations(data, col, constraints, profile_report) #pass report
                    if range_violations:
                          constraint_violations[col]["range_violation"] = range_violations

          if constraint_violations:
                  self.logger.debug(f"Numeric constraint validation failed: {constraint_violations}")
                  return ValidationResult(status=False, errors = constraint_violations)
          self.logger.info(f"Numeric constraint validation passed") 
          return ValidationResult(status=True)

     def _check_range_violations(self, data: pd.DataFrame, col: str, constraints: List[Dict[str, Any]], profile_report: Optional[Dict] = None) -> Optional[List[Any]]:
          """Helper method to check range violations for a specific column. Now also use the profile report to check for min/max"""
          
          range_constraints = next(
              (c for c in constraints if c["type"] == "range"), {}
          )
          min_value = range_constraints.get("min")
          max_value = range_constraints.get("max")

          # Pull min/max from profile report *if available* - overrides config if found in the report.
          if profile_report and col in profile_report.variables:
               if "min" in profile_report.variables[col]:
                    report_min = profile_report.variables[col]["min"]
                    if min_value is None or report_min < min_value:
                        min_value = report_min  # Use the report value if it's more restrictive
               if "max" in profile_report.variables[col]:
                    report_max = profile_report.variables[col]["max"]
                    if max_value is None or report_max > max_value:
                        max_value = report_max #Use report if more restrictive
          
          if min_value is not None or max_value is not None:
              mask = ~data[col].between(min_value, max_value, inclusive='both')
              if mask.any():
                  return data.loc[mask, col].tolist()
          return None

class DataValidation:
    def __init__(self, config: Configuration):
        self.config = config
        self.logger = logging.getLogger(__name__)  
        self.logger.info(f"Data Validation initialized")  
        self.logger.debug(f"Data validation config: {self.config}") 
        self.validators = [
            ColumnValidator(config),
            DataTypeValidator(config),
            MissingValueValidator(config),
            ConstraintValidator(config)
        ]
        
    def check_cardinality(self, data):
        """"Check  and drop columns with unique values"""

        unique_counts = data.nunique()
        drop_columns = [col for col in data.columns if unique_counts[col] == len(data)]
        if drop_columns: 
            logger.warning(f"Dropping columns with unique values: {drop_columns}")
    
        try:
            data.drop(columns=drop_columns, inplace=True)
            logger.debug(f"Dropped columns with unique values: {drop_columns}")
        except Exception as e:
            logger.error(f"Error occurred while dropping columns: {e}")
        
        return data
    
    def _generate_profile_report(self, data: pd.DataFrame) -> Dict:
        """
        Generates a pandas profile report and returns the path to the HTML report and also return the description.
        
        Args:
            data (pd.DataFrame): The DataFrame to profile
            
        Returns:
            Dict: the profile report and description.
        """
        try:
            report_path = Path(self.config.get_config().profile_report_path) #Get from config now
            
            profile = ProfileReport(data, title="Data Profiling Report")
            profile.to_file(str(report_path))
            description = profile.get_description() # Get the description
            
            self.logger.info(f"Profile report generated at: file://{report_path.absolute()}")
            return description
            
        except Exception as e:
            self.logger.error(f"Error generating or saving pandas profile report: {e}")
            raise CustomException(e, sys)
        
    # And update the validate_data method to use the returned path:
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate the data and return a status, save the metadata of the validation,
        and save the data if validation is passed
        """
        validation_results = {}
        overall_status = True
        profile_report = self._generate_profile_report(data) # Get report
        for validator in self.validators:
            result = validator.validate(data, profile_report)  #Pass the report
            validation_results[validator.__class__.__name__] = {"status": result.status, "errors": result.errors}
            if not result:
                overall_status = False
        

        # Validate or sanitize file paths
        val_status_path = Path(self.config.get_config().val_status).resolve(strict=False)
        root_dir_path = Path(self.config.get_config().root_dir).resolve(strict=False)
        
        # Save results to a file
        try:
            with open(val_status_path, 'w') as f:
                json.dump(validation_results, f, indent=4)
            logger.info(f"Validation results saved to {val_status_path}")
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")


        # Check and Drop columns with unique values
        self.check_cardinality(data)

        # Save the data to a parquet file only if the validation passed
        if overall_status:
            try:
                output_path = str(root_dir_path / 'validated_data.parquet')
                with open(output_path, 'wb') as f:
                    data.to_parquet(f, index=False)
                logger.info(f"Validated data saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save validated data: {e}")
        else:
            logger.warning(f"Data validation failed. Check {val_status_path} for more details")
        
        return overall_status
    

if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        config = config_manager.get_data_validation_config()
        data_validation = DataValidation(config=config)
        
        print(f"Data Directory: {config.get_config().data_dir}") #Check that the data directory is correct
        data = pd.read_parquet(config.get_config().data_dir)
        print("Loaded DataFrame Shape:", data.shape)
        
        logger.info("Starting data validation process") 
        validation_status = data_validation.validate_data(data)

        if validation_status:
            logger.info("Data Validation Completed Successfully!")
        else:
            logger.warning("Data Validation Failed. Check the status file for more details.")

    except Exception as e:
        logger.error(f"Data validation process failed: {e}")
        raise CustomException(e, sys)