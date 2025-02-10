import pandas as pd
import numpy as np
import joblib
import pathlib as Path

from dataclasses import dataclass
from typing import Any, Tuple, Dict

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
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
class ModelTrainerConfig:
    root_dir: Path
    train_features_path: Path
    train_targets_path: Path
    model_name: str
    model_params: Dict[str, Any]
    project_name: str
    random_state: int  
    val_features_path: Path
    val_targets_path: Path
    number_of_splits = int


class ConfigurationManager:
    def __init__(self,
                 model_training_config: Path = MODEL_TRAINER_CONFIG_FILEPATH,
                 model_params_config: Path = PARAMS_CONFIG_FILEPATH) -> None:

        try:
            self.training_config = read_yaml(model_training_config)
            self.model_params_config = read_yaml(model_params_config)
            create_directories([self.training_config.root_dir])

        except Exception as e:
            logger.error(f"Error in initializing ConfigurationManager")
            raise CustomException("Failed to initialize ConfigurationManager")

    def get_model_training_config(self) -> ModelTrainerConfig:
        logger.info("Getting model training configuration")
        try:
            trainer_config = self.training_config.model_trainer
            model_params = self.model_params_config.model_params

            # Create directories
            create_directories([trainer_config.root_dir])

            return ModelTrainerConfig(
                root_dir=Path(trainer_config.root_dir),
                train_features_path=Path(trainer_config.train_features_path),
                train_targets_path=Path(trainer_config.train_targets_path),
                model_name=trainer_config.model_name,
                model_params=model_params,
                project_name=trainer_config.project_name,
                random_state=trainer_config.random_state,  # Pass random_state to config
                val_features_path=Path(trainer_config.val_features_path),
                val_targets_path=Path(trainer_config.val_targets_path),
                number_of_splits=trainer_config.number_of_splits  
            )

        except Exception as e:
            logger.error(f"Error in getting model training configuration")
            raise CustomException("Failed to get model training configuration")


class DataManager:
    @staticmethod
    def load_data(features_path: Path, targets_path: Path) -> Tuple[Any, pd.Series]:
        with open(features_path, 'rb') as f:
            X_transformed = joblib.load(f)
        y = pd.read_parquet(targets_path.as_posix()).squeeze()
        return X_transformed, y

    @staticmethod
    def load_all_data(train_features_path: Path,
                      train_targets_path: Path,
                      val_features_path: Path,
                      val_targets_path: Path
                      ) -> Tuple[Any, pd.Series, Any, pd.Series]:
        try:
            X_train_transformed, y_train = DataManager.load_data(train_features_path, train_targets_path)
            X_val_transformed, y_val = DataManager.load_data(val_features_path, val_targets_path)

            return X_train_transformed, y_train, X_val_transformed, y_val

        except FileNotFoundError as fnf_error:
            logger.error(f"File not found: {str(fnf_error)}")
            raise CustomException(fnf_error, sys)
        except Exception as e:
            logger.error(f"Unexpected error loading data: {str(e)}")
            raise CustomException(e, sys)


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig) -> None:
        self.config = config
        self.X_train_transformed, self.y_train, self.X_val_transformed, self.y_val = self.load_and_prepare_data()

    def load_and_prepare_data(self) -> Tuple[Any, pd.Series, Any, pd.Series]:
        """Loads and prepares the training and validation data."""
        try:
            data_manager = DataManager()
            X_train_transformed, y_train, X_val_transformed, y_val = data_manager.load_all_data(
                self.config.train_features_path,
                self.config.train_targets_path,
                self.config.val_features_path,
                self.config.val_targets_path,
            )
            logger.info("Data loaded and prepared successfully within ModelTrainer.")

            return X_train_transformed, y_train, X_val_transformed, y_val

        except Exception as e:
            logger.error(f"Error loading and preparing data: {str(e)}")
            raise CustomException(e, sys)

    def train(self): #No parameters needed here
        try:
            if not self.config.model_params:
                raise ValueError("Model parameters not provided.")

            run = wandb.init(
                project=self.config.project_name,
                config={**self.config.model_params, "random_state": self.config.random_state}  
            )
            gb_model = GradientBoostingRegressor(**self.config.model_params,
                                                 random_state=self.config.random_state)  

            # Perform K-Fold Cross validation
            kf = KFold(n_splits=self.config.number_of_splits, shuffle=True, random_state=self.config.random_state)  

            cv_scores = cross_val_score(
                gb_model, self.X_train_transformed, self.y_train, cv=kf,
                scoring="neg_root_mean_squared_error"
            )  # Use 'neg_root_mean_squared_error'
            
            cv_rmse_scores = -cv_scores  # Convert back to positive RMSE
            mean_cv_rmse = np.mean(cv_rmse_scores)
            std_cv_rmse = np.std(cv_rmse_scores)

            logger.info(f"K-Fold Cross-validation RMSE scores: {cv_rmse_scores}")
            logger.info(f"Mean K-Fold cross-validation RMSE: {mean_cv_rmse}")
            logger.info(f"Standard deviation of K-Fold cross-validation RMSE: {std_cv_rmse}")

            wandb.log({"mean_cv_rmse": mean_cv_rmse, "std_cv_rmse": std_cv_rmse})

            # Fit on the entire training set after cross-validation
            gb_model.fit(self.X_train_transformed, self.y_train)

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

            # Save model
            model_path = Path(self.config.root_dir) / self.config.model_name
            joblib.dump(gb_model, model_path)
            logger.info(f"Model trained and saved at: {model_path}")

            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(model_path)
            run.log_artifact(artifact)

            run.finish()

            return gb_model
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        model_training_config = config_manager.get_model_training_config()
        model_trainer = ModelTrainer(config=model_training_config) 

        # Train the model
        model = model_trainer.train() #
        logger.info("Model Training Completed Successfully")

    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        wandb.finish()
        sys.exit(1)