# Project : mlops-logistic-regression
Train a Logistic Regression model, track experiments with MLflow, version data and models with DVC, containerize the model for serving with Docker, and deploy it scalably on Kubernetes.

## Prerequisites:
- Python 3.8+
- Git
- Docker Desktop (includes Docker Engine and Kubernetes for local setup) or access to a Kubernetes cluster.
- Minikube (for local Kubernetes cluster setup, if not using Docker Desktop's built-in K8s)
- kubectl (Kubernetes command-line tool)
- Helm (for easier Kubernetes deployments, optional but recommended)

## Setup In local Machine 
- Open two terminal 

## Go to project directory
  - cd mlops-logistic-regression
  
## Create python environment
  - python -m venv venv
  - .\venv\Scripts\activate.bat
  
## Install all required dependencies
- pip install -r requirements.txt

## prepare model traning data
- python src/data_preparation.py

## Setup DVC(Data Version Control) with Git
- git init
- dvc init --no-scm
- git add .
- git commit -m "Initial project setup and data generation script"

- dvc add data/raw/iris.csv
- git add data/raw/iris.csv.dvc .gitignore
- git commit -m "Versioned iris.csv with DVC"

- Note : Create a directory outside your project for "dvc_cache"
- mkdir ../dvc_cache
- dvc remote add -d my_local_remote ../dvc_cache
- dvc push
- git add .dvc/config
- git commit -m "Configured local DVC remote and pushed data"


## Mlflow run in seconod terminal with same directory "mlops-logistic-regression"
- mlflow ui --port 5000
- Test Mlflow : http://127.0.0.1:5000/

## Run traning to Model
- Note : After traning go to this location and copy the "model.pkl" into directory "mlops-logistic-regression"
- D:\mlops-project\mlops-logistic-regression\mlartifacts\0\models\m-13c33b1afe0b4fab9c5185e9593cbcc6\artifacts\model.pkl

- python src/train_model.py

## Build docker image
- Note: Before build docker image change the path with newly created after traning model
- COPY mlartifacts/0/models/m-13c33b1afe0b4fab9c5185e9593cbcc6/artifacts/model.pkl .
- docker build -t logistic-regression-model:latest .

## Run docker container
- YOUR_MLFLOW_RUN_ID : take from mlflow server
- docker run -p 5000:5000 -e MLFLOW_MODEL_URI="runs:/YOUR_MLFLOW_RUN_ID/logistic_regression_model" logistic-regression-model:latest

## Test Running Docker Application working or not
- curl -X POST -H "Content-Type: application/json" -d '[[5.1, 3.5, 1.4, 0.2]]' http://localhost:5000/predict

## kubernatics deployment 
- Note : change MLFLOW_MODEL_URI value with latest runid
- kubectl apply -f kubernetes/deployment.yaml
- kubectl apply -f kubernetes/service.yaml

- kubectl get deployments
- kubectl get pods
- kubectl get services
- kubectl delete service <service-name>
- kubectl delete pod <pod-name>
- kubectl deployments pod <pod-name>

## Testing kubernatics deployment Application
- curl -X POST -H "Content-Type: application/json" -d '[[5.1, 3.5, 1.4, 0.2]]' http://localhost:80/predict 

## Some screen shot for reference
![image](https://github.com/user-attachments/assets/0a81b5c9-6f7f-4e25-aa79-056b18bef3f9)
