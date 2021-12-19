FROM python:3.8

# ENV SERVER_HOST 0.0.0.0
# ENV SERVER_PORT 5000

# COPY . /flask-app
# WORKDIR /flask-app
RUN pip install mlflow==1.22.0

EXPOSE 5050

ENTRYPOINT mlflow server --host 0.0.0.0 --port 5050 --backend-store-uri sqlite:///ml.db --default-artifact-root ./mlartifacts