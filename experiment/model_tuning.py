
import sys
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error  

# Local Modules
from src.ElectricityBill.exception import CustomException
from src.ElectricityBill.logger import logger
from src.ElectricityBill.constants import *
from src.ElectricityBill.utils.commons import *

# Wandb
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
    val_features_path: Path
    val_targets_path: Path


class ConfigurationManager:
    def __init__(
        self,
        model_training_config: Path = MODEL_TRAINER_CONFIG_FILEPATH,
        model_params_config: Path = PARAMS_CONFIG_FILEPATH,
        hyperparameter_config: Path = HYPERPARAMETER_SEARCH_CONFIG_FILEPATH,
    ):
        try:
            self.training_config = read_yaml(model_training_config)
        except Exception as e:
            logger.error(f"Error loading model training config file: {str(e)}")
            raise CustomException(e, sys)

        try:
            self.model_params_config = read_yaml(model_params_config)
            self.wandb_config = read_yaml(hyperparameter_config)  # Store the entire WandB config
        except Exception as e:
            logger.error(f"Error loading model parameters config: {str(e)}")
            raise CustomException(e, sys)

        try:
            if "artifacts_root" in self.training_config:
                artifacts_root = self.training_config.artifacts_root
                create_directories([artifacts_root])
            else:
                logger.error("artifacts_root not defined in the configuration.")
                raise CustomException(
                    "artifacts_root not defined in the configuration.", sys
                )
        except Exception as e:
            logger.error(f"Error creating directories: {str(e)}")
            raise CustomException(e, sys)

    def get_model_training_config(self) -> ModelTrainerConfig:
        logger.info("Getting model training configuration")
        try:
            trainer_config = self.training_config["model_trainer"]
            model_params = self.model_params_config["GradientBoostingRegressor"]  

            # Creates all necessary directories
            create_directories([trainer_config.root_dir])

            return ModelTrainerConfig(
                root_dir=Path(trainer_config.root_dir),
                train_features_path=Path(trainer_config.train_features_path),
                train_targets_path=Path(trainer_config.train_targets_path),
                model_name=trainer_config.model_name,
                model_params=model_params,
                project_name=trainer_config.project_name,
                val_features_path=Path(trainer_config.val_features_path),
                val_targets_path=Path(trainer_config.val_targets_path),
            )
        except Exception as e:
            logger.error(f"Error getting model training config: {str(e)}")
            raise CustomException(e, sys)


class DataManager:
    @staticmethod
    def load_data(
        train_features_path: Path,
        train_targets_path: Path,
        val_features_path: Path,
        val_targets_path: Path,
    ) -> Tuple[Any, pd.DataFrame, pd.DataFrame, Any]:
        """Loads the training and validation data from the given file paths."""
        try:
            with open(train_features_path, "rb") as f:
                X_train_transformed = joblib.load(f)
            y_train = pd.read_parquet(train_targets_path.as_posix())

            with open(val_features_path, "rb") as f:
                X_val_transformed = joblib.load(f)
            y_val = pd.read_parquet(val_targets_path.as_posix())

            logger.info("Training and validation data loaded successfully")
            return X_train_transformed, y_train, y_val, X_val_transformed

        except FileNotFoundError as fnf_error:
            logger.error(f"File not found: {str(fnf_error)}")
            raise CustomException(fnf_error, sys)
        except Exception as e:
            logger.error(f"Unexpected error loading data: {str(e)}")
            raise CustomException(e, sys)


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def _train_and_evaluate(
        self, X_train_transformed, y_train, X_val_transformed, y_val, sweep_configuration
    ):
        with wandb.init() as run:
            config = wandb.config
            gb_model = GradientBoostingRegressor(**config)  
            gb_model.fit(X_train_transformed, y_train.values.ravel()) 
            y_val_pred = gb_model.predict(X_val_transformed)
            mse = mean_squared_error(y_val, y_val_pred) 
            wandb.log({"validation_mse": mse})

    def train_with_sweep(
      self,
      X_train_transformed,
      y_train,
      X_val_transformed,
      y_val,
      sweep_configuration,
    ):

        sweep_id = wandb.sweep(
            sweep=sweep_configuration, project=self.config.project_name
        )
        wandb.agent(
            sweep_id,
            function=lambda: self._train_and_evaluate(
                X_train_transformed, y_train, X_val_transformed, y_val, sweep_configuration
            ),
            count=10,  # Or specify a different number of runs
        )


if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        model_training_config = config_manager.get_model_training_config()
        data_manager = DataManager()
        X_train_transformed, y_train, y_val, X_val_transformed = data_manager.load_data(
            model_training_config.train_features_path,
            model_training_config.train_targets_path,
            model_training_config.val_features_path,
            model_training_config.val_targets_path,
        )
        model_trainer = ModelTrainer(config=model_training_config)
        sweep_config = config_manager.wandb_config.get("sweep", {})

        model_trainer.train_with_sweep(
            X_train_transformed,
            y_train,
            X_val_transformed,
            y_val,
            sweep_config
        )

    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        wandb.finish()  # Ensure the run is ended in case of an exception
        sys.exit(1)