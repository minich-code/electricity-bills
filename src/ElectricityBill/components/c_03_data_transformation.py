

import pandas as pd
import joblib
import sys

from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.ElectricityBill.exception import CustomException
from src.ElectricityBill.logger import logger
from src.ElectricityBill.utils.commons import *
from src.ElectricityBill.constants import *
from src.ElectricityBill.config_entity.config_params import DataTransformationConfig



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_transformer_object(self) -> ColumnTransformer:
        logger.info("Creating a transformer object")

        try:
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())

            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, self.config.numerical_cols),
                    ('cat', categorical_transformer, self.config.categorical_cols),
                ], remainder='passthrough'
            )

            return preprocessor

        except Exception as e:
            logger.error(f"Error creating transformer object: {e}")
            raise CustomException(e)

    def train_val_test_split(self):
        logger.info("Splitting data into train, validation and test sets")

        try:
            df = pd.read_parquet(self.config.data_path)
            logger.info(f"Data loaded from {self.config.data_path}")

            # Define the features and target variable
            X = df.drop(columns=[self.config.target_col])
            y = df[self.config.target_col]

            # Split into training and temporary sets
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, random_state=self.config.random_state
            )

            # Split the temporary set into validation and test sets
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=self.config.random_state
            )

            logger.info("Data split successfully")

            # Save the target variables
            y_train.to_frame().to_parquet(self.config.root_dir / 'y_train.parquet', index=False)
            y_val.to_frame().to_parquet(self.config.root_dir / 'y_val.parquet', index=False)
            y_test.to_frame().to_parquet(self.config.root_dir / 'y_test.parquet', index=False)

            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self,
                                     X_train: pd.DataFrame, X_val: pd.DataFrame,
                                     X_test: pd.DataFrame, y_train: pd.Series,
                                     y_val: pd.Series, y_test: pd.Series
                                     ) -> Tuple[ColumnTransformer, pd.DataFrame, pd.Series]:

        try:
            # Get the transformer object
            preprocessor_obj = self.get_transformer_object()

            if not isinstance(preprocessor_obj, ColumnTransformer):
                raise TypeError("Invalid transformer object")

            # Transform the training, testing and validation data
            X_train_transformed = preprocessor_obj.fit_transform(X_train)
            X_val_transformed = preprocessor_obj.transform(X_val)
            X_test_transformed = preprocessor_obj.transform(X_test)

            # Save the transformed data
            X_train_transformed_path = self.config.root_dir / "X_train_transformed.joblib"  # Construct Path objects directly
            X_val_transformed_path = self.config.root_dir / "X_val_transformed.joblib"
            X_test_transformed_path = self.config.root_dir / "X_test_transformed.joblib"

            joblib.dump(X_train_transformed, X_train_transformed_path)
            joblib.dump(X_val_transformed, X_val_transformed_path)
            joblib.dump(X_test_transformed, X_test_transformed_path)

            # Save the preprocessing object
            preprocessor_path = Path(self.config.root_dir) / "preprocessor.joblib"
            save_object(obj=preprocessor_obj, file_path=preprocessor_path)

            return preprocessor_obj, X_train_transformed, X_val_transformed, X_test_transformed, y_train, y_val, y_test


        except Exception as e:
            logger.error(f"Error initiating data transformation: {e}")
            raise CustomException(e, sys)

