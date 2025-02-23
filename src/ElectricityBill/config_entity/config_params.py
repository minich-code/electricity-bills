

from dataclasses import dataclass 
from pathlib import Path
from typing import Dict, List, Any



# Data ingestion configuration
@dataclass
class DataIngestionConfig:
    root_dir: str
    database_name: str
    collection_name: str
    batch_size: int
    mongo_uri: str


# Data validation 
@dataclass
class DataValidationConfig:
    root_dir: str
    data_dir: str
    val_status: str
    all_schema: dict
    validated_data: str
    profile_report_name: str


# Data Transformation
@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    random_state: frozenset
    target_col: frozenset
    numerical_cols: List[str]
    categorical_cols: List[str]

