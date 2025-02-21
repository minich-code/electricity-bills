[tool.poetry]
name = "electricity_bills"
version = "0.0.1"
description = "This is a project to predict electricity bills for different households"
authors = ["Your Name <your.email@example.com>"]
license = "MIT"
readme = "README.md"
keywords = ["electricity price", "machine-learning", "prediction price"]
packages = [{include = "ElectricityBill", from = "src"}]

[tool.poetry.dependencies]
python = ">=3.10,<4.0"
ensure = "^1.0.0"
flask = "^2.0.0"
Flask-Cors = "^3.0.0"
joblib = "^1.0.0"
matplotlib = "^3.0.0"
numpy = "^1.20.0"
pandas = "^1.3.0"
pymongo = "^4.0.0"
PyYAML = "^6.0"
scikit-learn = "^1.0.0"
seaborn = "^0.11.0"
streamlit = "^1.0.0"
types-PyYAML = "^6.0"
python-box = "^6.0"
ydata-profiling = "^5.0"
pydantic = "^2.0.0"
pydantic-settings = "^2.0.0"
dvc = "^3.0.0"
wandb = "^0.15.0"
apache-airflow = "^2.0.0"
python-dotenv = "^0.21.0"
apache-airflow-providers-mongo = "^4.0.0"
fastapi = "^0.70.0"

[tool.poetry.group.dev.dependencies]
flake8 = "^4.0.0"
pytest = "^7.0.0"

[tool.poetry.scripts]
electricity_bills = "electricity_bills.main:main"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"