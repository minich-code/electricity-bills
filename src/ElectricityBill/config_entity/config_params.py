from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

# Data Ingestuion config 




# Model trainer 
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

