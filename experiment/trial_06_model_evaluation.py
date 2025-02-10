import sys

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import wandb

from src.ElectricityBill.exception import CustomException
from src.ElectricityBill.logger import logger
from src.ElectricityBill.constants import *
from src.ElectricityBill.utils.commons import *
wandb.require("core")


@dataclass
class ModelValidationConfig:
    root_dir: Path
    test_feature_path: Path
    test_targets_path: Path
    model_path: Path
    validation_scores_path: Path
    project_name: str


class ConfigurationManager:
    def __init__(self, model_validation_config: str = MODEL_VALIDATION_CONFIG_FILEPATH):
        self.validation_config = read_yaml(model_validation_config)
        artifacts_root = self.validation_config.artifacts_root
        create_directories([artifacts_root])

    def get_model_validation_config(self) -> ModelValidationConfig:
        logger.info("Getting model validation configuration")

        val_config = self.validation_config.model_validation
        create_directories([val_config.root_dir])

        return ModelValidationConfig(
            root_dir=Path(val_config.root_dir),
            test_feature_path=Path(val_config.test_feature_path),
            test_targets_path=Path(val_config.test_targets_path),
            model_path=Path(val_config.model_path),
            validation_scores_path=Path(val_config.validation_scores_path),
            project_name = val_config.project_name
        )


class ModelValidation:
    def __init__(self, config: ModelValidationConfig):
        self.config = config
        self.model = None  # Initialize model attribute
        self.X_test = None
        self.y_test = None
        self.run = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Logs the start of model validation data loading and retrieves
        paths for test features and test targets from the configuration.
        """
        logger.info("Loading model validation data")
        try:
            test_features = self.config.test_feature_path
            test_targets = self.config.test_targets_path

            X_test = joblib.load(test_features)
            y_test = pd.read_parquet(test_targets)

            # Convert y_test to a Series if it's a DataFrame
            if isinstance(y_test, pd.DataFrame):
                y_test = y_test.squeeze()

            logger.info("Model validation data loaded successfully")
            return X_test, y_test

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException(f"Error loading data: {e}")

    def load_model(self) -> object:
        logger.info("Loading model")
        try:
            model = joblib.load(self.config.model_path)
            logger.info("Model loaded successfully")
            return model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise CustomException(f"Error loading model: {e}")

    def validate_model(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Validates the loaded model on the test data and logs metrics.
        """
        logger.info("Validating the model on the test data")

        try:
            # Make predictions
            y_pred = self.model.predict(X_test)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            # Calculate Adjusted R-squared
            n = X_test.shape[0]  # Number of samples
            p = X_test.shape[1]  # Number of features
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

            # Log metrics to W&B
            metrics = {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
                "Adjusted R2": adjusted_r2,
                "MAPE": mape
            }

            logger.info(f"Validation metrics: {metrics}")

            # Save metrics to JSON file
            with open(self.config.validation_scores_path, 'w') as f:
                json.dump(metrics, f, indent=4)  # Add indent for readability

            # Log metrics to W&B if run is initialized
            if self.run:
                self.run.log(metrics)

        except Exception as e:
            logger.error(f"Error during model validation: {e}")
            raise CustomException(f"Error during model validation: {e}", error_details=str(e))

    def run_validation(self):
        """
        Runs the complete model validation process.
        """
        try:
            # Initialize W&B run
            self.run = wandb.init(project=self.config.project_name, config=self.config.__dict__)

            X_test, y_test = self.load_data()
            model = self.load_model()
            self.model = model
            self.validate_model(X_test, y_test)  # Pass data and model to the validation function


        except Exception as e:
            logger.error(f"Error during model validation: {e}")
            raise CustomException(f"Error during model validation: {e}", error_details=str(e))

        finally:
            if self.run:
                self.run.finish()


if __name__ == "__main__":
    try:
        config = ConfigurationManager()
        model_validation_config = config.get_model_validation_config()
        model_validation = ModelValidation(config=model_validation_config)
        model_validation.run_validation()
    except Exception as e:
        logger.error(f"Error during the model validation main process: {e}")
        raise CustomException(f"Error during the model validation main process: {e}")