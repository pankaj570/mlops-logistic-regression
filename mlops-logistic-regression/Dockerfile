# Dockerfile
# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/predict.py .
COPY mlartifacts/0/models/m-13c33b1afe0b4fab9c5185e9593cbcc6/artifacts/model.pkl .
# If you also need data_preparation.py in the container for some reason, uncomment this:
# COPY src/data_preparation.py . 

# Expose the port our Flask app runs on
EXPOSE 5000

# Set the MLFLOW_MODEL_URI environment variable.
# This value will be overridden by Kubernetes deployment.yaml for actual deployment.
# For local Docker testing, you must provide it via -e flag.
#ENV MLFLOW_MODEL_URI="runs:/dummy_run_id/logistic_regression_model"

# Command to run the Flask application using Gunicorn
# The 'predict:app' refers to the 'app' object in 'predict.py'
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "predict:app"]