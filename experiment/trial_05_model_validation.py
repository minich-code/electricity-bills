
import sys
sys.path.append('/home/western/ds_projects/electricity-bills')
import pandas as pd
import numpy as np
import joblib
import pathlib as Path
import os  # Import os module

from dataclasses import dataclass
from typing import Any, Tuple, Dict

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Custom modules
from src.ElectricityBill.exception import CustomException
from src.ElectricityBill.logger import logger
from src.ElectricityBill.constants import *
from src.ElectricityBill.utils.commons import *

# Weights and biases
import wandb
wandb.require("core")

@dataclass
class ModelValidationConfig:
    root_dir: Path
    val_features_path: Path
    val_targets_path: Path
    model_path: Path  # Path to the PRE-TRAINED model
    project_name: str
    random_state: int


class ConfigurationManager:
    def __init__(self,
                 model_validation_config_path: Path = MODEL_VALIDATION_CONFIG_FILEPATH) -> None:

        try:
            self.model_validation_config = read_yaml(model_validation_config_path)
            create_directories([self.model_validation_config['artifacts_root']])

        except Exception as e:
            logger.error(f"Error in initializing ConfigurationManager: {str(e)}")
            raise CustomException(e, sys)

    def get_model_validation_config(self) -> ModelValidationConfig:
        logger.info("Getting model validation configuration")
        try:
            model_val_config = self.model_validation_config['model_validation']  # Correct key name
            # Create directories
            create_directories([model_val_config['root_dir']])

            return ModelValidationConfig(
                root_dir=Path(model_val_config['root_dir']),
                val_features_path=Path(model_val_config['val_features_path']),
                val_targets_path=Path(model_val_config['val_targets_path']),
                model_path=Path(model_val_config['model_path']),
                project_name=model_val_config['project_name'],
                random_state=int(model_val_config['random_state'])
            )

        except Exception as e:
            logger.error(f"Error in getting model validation configuration: {str(e)}")
            raise CustomException(e, sys)


class DataManager:
    @staticmethod
    def load_data(features_path: Path, targets_path: Path) -> Tuple[Any, pd.Series]:
        with open(features_path, 'rb') as f:
            X_transformed = joblib.load(f)
        y = pd.read_parquet(targets_path.as_posix()).squeeze()
        return X_transformed, y

    @staticmethod
    def load_validation_data(val_features_path: Path,
                      val_targets_path: Path
                      ) -> Tuple[Any, pd.Series]:
        try:

            X_val_transformed, y_val = DataManager.load_data(val_features_path, val_targets_path)

            return X_val_transformed, y_val

        except FileNotFoundError as fnf_error:
            logger.error(f"File not found: {str(fnf_error)}")
            raise CustomException(fnf_error, sys)
        except Exception as e:
            logger.error(f"Unexpected error loading data: {str(e)}")
            raise CustomException(e, sys)


class ModelValidator:
    def __init__(self, config: ModelValidationConfig) -> None:
        self.config = config
        self.X_val_transformed, self.y_val = self.load_and_prepare_data()

    def load_and_prepare_data(self) -> Tuple[Any, pd.Series]:
        """Loads and prepares the validation data."""
        try:
            data_manager = DataManager()
            X_val_transformed, y_val = data_manager.load_validation_data(
                self.config.val_features_path,
                self.config.val_targets_path,
            )
            logger.info("Validation data loaded and prepared successfully.")

            return X_val_transformed, y_val

        except Exception as e:
            logger.error(f"Error loading and preparing validation data: {str(e)}")
            raise CustomException(e, sys)

    def load_model(self):
        """Loads the pre-trained model."""
        try:
            model_path = self.config.model_path
            gb_model = joblib.load(model_path)
            logger.info(f"Loaded pre-trained model from: {model_path}")
            return gb_model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise CustomException(e, sys)

    def validate(self, run_number: int):
        try:

            # Load pre-trained model
            gb_model = self.load_model()

            # Initialize WandB run with a dynamic run name
            run_name = f"Validation {run_number}"
            run = wandb.init(
                project=self.config.project_name,
                name=run_name,
                config={"random_state": self.config.random_state}
            )

            # Evaluate on validation set
            y_val_pred = gb_model.predict(self.X_val_transformed)

            # Calculate Metrics
            mae = mean_absolute_error(self.y_val, y_val_pred)
            mse = mean_squared_error(self.y_val, y_val_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_val, y_val_pred)

            # Calculate Adjusted R-squared
            n = len(self.y_val)  # Number of samples
            p = self.X_val_transformed.shape[1]  # Number of features
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

            # MAPE (Mean Absolute Percentage Error) - An important metric for regression
            def calculate_mape(y_true, y_pred):
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            mape = calculate_mape(self.y_val, y_val_pred)

            # Log Metrics to WandB
            wandb.log({
                "validation_mae": mae,
                "validation_mse": mse,
                "validation_rmse": rmse,
                "validation_r2": r2,
                "validation_adjusted_r2": adjusted_r2,
                "validation_mape": mape  # Log MAPE
            })

            logger.info(f"Validation metrics logged to WandB run: {run_name}")

            run.finish()

        except Exception as e:
            logger.error(f"Error during model validation: {str(e)}")
            raise CustomException(e, sys)


def get_run_count_from_file(root_dir: str, filename="run_count.txt") -> int:
    """
    Reads the current run count from a single file.
    If the file doesn't exist, returns 0.
    """
    filepath = os.path.join(root_dir, filename)
    try:
        with open(filepath, 'r') as f:
            count = int(f.read().strip())
        return count
    except FileNotFoundError:
        return 0
    except ValueError:
        logger.warning("Run count file is corrupted. Resetting to 0.")
        return 0  # Handle case where the file contains non-integer data


def write_run_count_to_file(root_dir: str, count: int, filename="run_count.txt") -> None:
    """
    Writes the new run count to the single tracking file.
    """
    filepath = os.path.join(root_dir, filename)
    try:
        with open(filepath, 'w') as f:
            f.write(str(count))
    except Exception as e:
        logger.error(f"Error writing run count to file: {str(e)}")


if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        model_validation_config = config_manager.get_model_validation_config()
        model_validator = ModelValidator(config=model_validation_config)

        # Determine next run number
        root_dir = model_validation_config.root_dir
        run_number = get_run_count_from_file(root_dir) + 1

        # Validate the model
        model_validator.validate(run_number)

        # Write the updated run number back to the file
        write_run_count_to_file(root_dir, run_number)

        logger.info("Model Validation Completed Successfully")

    except Exception as e:
        logger.error(f"Error in model validation: {str(e)}")
        wandb.finish()
        sys.exit(1)