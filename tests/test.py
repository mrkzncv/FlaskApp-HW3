import importlib
test_app = importlib.import_module("flask-app.app")
# from calc import app as test_app
import unittest
import numpy as np
from sklearn.datasets import make_classification, make_regression

X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                            random_state=0, shuffle=True)

Z, r = make_regression(n_samples=1000, n_features=10,
                           n_informative=4, random_state=0, shuffle=True)

# Валидация схемы данных: формат данных: csv - splitter
# Кол-во записей - shape?
# пропуски,
# Наличие необходимых полей
# типы данных

class TestData(unittest.TestCase):
    def setUp(self):
        self.client = test_app.app.test_client()

    def test_get(self):
        response = self.client.get('/api/ml_models')
        self.assertIn(b'classification', response.data) # ищем в строке
        self.assertEqual(response.status_code, 200)

    def test_post(self):
        response = self.client.post('/api/ml_models',
                         json={'problem': 'classification',
                               'name': 'SVM',
                               'h_tune': False,
                               'X':X.tolist(), 'y':y.tolist()})
        self.assertEqual(response.status_code, 200)

    def test_unknown_model(self):
        response = self.client.post('/api/ml_models',
                                    json={'problem': 'clustering',
                                          'name': 'SVM',
                                          'h_tune': False,
                                          'X': X.tolist(), 'y': y.tolist()})
        self.assertEqual(response.status_code, 500)