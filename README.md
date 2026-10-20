cat > README.md <<'EOF'
# MLOps Customer Churn Prediction

A machine learning API that predicts whether a telecom customer is
likely to churn. Built with scikit-learn, MLflow, FastAPI, and Docker.

## Features

- Customer churn prediction using a trained ML pipeline
- MLflow experiment tracking and model artifacts
- FastAPI REST API with interactive Swagger documentation
- Docker containerization
- Automated API tests with pytest
- GitHub Actions CI planned for automated test execution

## Project Structure

- `src/train.py` - Model training and experiment tracking
- `src/api.py` - FastAPI prediction service
- `tests/test_api.py` - Automated API tests
- `models/churn_pipeline/` - Exported trained model
- `Dockerfile` - Docker image configuration
- `.github/workflows/` - CI workflow

## Run Locally

Create and activate a Python virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt