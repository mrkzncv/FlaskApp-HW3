FROM python:3.8

ENV HOST 0.0.0.0
ENV PORT_MLFLOW 5050
ENV PORT 8080
# ENV LC_ALL en_US.utf-8
# ENV LANG en_US.utf-8

COPY . /flask-app
WORKDIR /flask-app
RUN pip install -r requirements.txt
RUN export DEBUG_METRICS=false

EXPOSE 8080

CMD ["python3", "cli.py"]