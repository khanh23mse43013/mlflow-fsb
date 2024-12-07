
my_env_init:
	python3 -m venv my-venv && source my-venv/bin/activate

run_mlflow:
	python3 -m mlflow ui --port 5000

install:
	pip3 install -r requirements.txt

run:
	uvicorn main:app --host 127.0.0.1 --port 3000 --reload

train_model:
	python3 src/trains/train_model.py

mlflow_server:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host
