from flask import Flask
from flask_restx import Api
from .ml_models import MLModelsDAO
from prometheus_flask_exporter import PrometheusMetrics
# from log import log
import logging

logging.basicConfig(
    filename='logs/app.log',
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    # handlers = handlers,
    datefmt='%d-%b-%y %H:%M:%S'
)

logging.info('This is an info message')
logging.error('This is an error message')

log = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app)

models_dao = MLModelsDAO()
metrics = PrometheusMetrics(app)

from app import views