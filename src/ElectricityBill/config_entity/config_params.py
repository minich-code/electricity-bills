

from dataclasses import dataclass 
from pathlib import Path
from typing import Dict, List, Any



# Data ingestion configuration
@dataclass
class DataIngestionConfig:
    config_data: dict


# Data validation 
@dataclass
class DataValidationConfig:
    root_dir: Path
    val_status: str
    data_dir: Path
    all_schema: Dict[str, Any]
    critical_columns: List[str]
    profile_report_path: str  # Add path for the profile report

# Data Transformation
@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    random_state: frozenset
    target_col: frozenset
    numerical_cols: List[str]
    categorical_cols: List[str]

