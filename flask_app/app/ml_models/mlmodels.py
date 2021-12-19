import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV
import os

PORT =  os.environ['PORT_MLFLOW']
# HOST = 0.0.0.0
# PORT = 5000

import mlflow.sklearn
from mlflow.models.signature import infer_signature
import mlflow.pyfunc

from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(f'http://mlflow:{PORT}/')
client = MlflowClient()

class MLModelsDAO:
    def __init__(self, ):
        self.ml_models = {'data': []}
        self.counter = 0

    def get(self, id, log):
        """
        Функция обучает модель с заданным id и выдает предсказания
        :param id: integer: уникальный идентификатор модели
        :return: list: предсказания модели
        """
        f_name = None
        for model in self.ml_models['data']:
            if model['id'] == id:
                # выгрузить модель с последней версией
                model_name = f"{model['problem']}_{model['name']}_{model['id']}"
                all_versions = client.search_model_versions("name='{}'".format(model_name))
                last_experiment = 1
                for v in all_versions:
                    if int(v.version) >= last_experiment:
                        last_experiment = int(v.version)
                trained_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{last_experiment}")
                prediction = trained_model.predict(np.array(model['X']))
                # f_name = f"models static/{model['problem']}_{model['name']}_{model['id']}.pickle"
                # trained_model = pickle.load(open(f_name, 'rb'))
                # prediction = trained_model.predict(np.array(model['X']))
                log.info(f'ml_model {model["id"]} %s predictions successfully calculated')
                return prediction.tolist()
        if f_name is None:
            log.error(f'%s failed to predict, ml_model {id} does not exist')
            raise NotImplementedError('ml_model {} does not exist'.format(id))

    def create(self, data, log, is_new=True):  # пришли данные, надо присвоить id (для POST)
        """
        Обучение (переобучение) модели. На вход подается запрос на обучение модели и данные.
        Если у нас предусмотрена запрашиваемая функциональность, модель обучается и записывается в pickle
        с id модели в названии файла. Путь до файла записывается в json с ключом 'model_path'.
        :param data: json {'problem': 'classification', 'name': 'Random Forest', 'h_tune': False, 'X':x, 'y':y}
        :param is_new: boolean: новая ли модель или надо переобучать существующую
        :return: список обученных моделей
        """
        ml_model = data
        if (ml_model['problem'] in ['classification', 'regression']) and \
                (ml_model['name'] in ['Random_Forest', 'SVM']):
            if is_new:
                self.counter += 1
                ml_model['id'] = self.counter
            else: # удаляем все версии старой модели
                registered_model_name = f"{ml_model['problem']}_{ml_model['name']}_{ml_model['id']}"
                all_versions = client.search_model_versions("name='{}'".format(registered_model_name))
                for version in all_versions:
                    mlflow.delete_run(version.run_id)
            x, y = np.array(ml_model['X']), np.array(ml_model['y']) # распаковать в таске
            if ml_model['problem'] == 'classification':
                best_model = classification(ml_model['name'], x, y, h_tune=ml_model['h_tune'])  # обучение
                mlflow.set_experiment(f"classifier")
                with mlflow.start_run(run_name=f'{ml_model["problem"]}_{ml_model["name"]}'):
                    mlflow.log_params(best_model.get_params())  # возвращает словарь
                    mlflow.log_metrics({'auc': best_model.score(x, y)})
                    signature = infer_signature(x, best_model.predict(x))
                    mlflow.sklearn.log_model(
                                    best_model, 'skl_model',
                                    signature = signature,
                                    registered_model_name =
                                          f"{ml_model['problem']}_{ml_model['name']}_{ml_model['id']}")
                # f_name = f"models static/{ml_model['problem']}_{ml_model['name']}_{ml_model['id']}.pickle"
                # pickle.dump(best_model, open(f_name, 'wb'))
                # ml_model['model_path'] = f_name
            elif ml_model['problem'] == 'regression':
                best_model = regression(ml_model['name'], x, y, h_tune=ml_model['h_tune']) #обучение
                mlflow.set_experiment(f"regressor")
                with mlflow.start_run(run_name=f'{ml_model["problem"]}_{ml_model["name"]}'):
                    mlflow.log_params(best_model.get_params())  # возвращает словарь
                    mlflow.log_metrics({'R2': best_model.score(x, y)})
                    signature = infer_signature(x, best_model.predict(x))
                    mlflow.sklearn.log_model(
                                    best_model, 'skl_model',
                                    signature = signature,
                                    registered_model_name =
                                          f"{ml_model['problem']}_{ml_model['name']}_{ml_model['id']}")
                # f_name = f"models static/{ml_model['problem']}_{ml_model['name']}_{ml_model['id']}.pickle"
                # pickle.dump(best_model, open(f_name, 'wb'))
                # ml_model['model_path'] = f_name
            if is_new:
                self.ml_models['data'].append(ml_model)
                log.info(f'ml_model {ml_model["id"]} %s successfully added')
        else:
            log.error(f'%s failed to create - unsupported model name {ml_model["name"]} or problem {ml_model["problem"]}')
            raise NotImplementedError("""Сейчас доступны для обучения только classification and regression:
                                        Random Forest или SVM""")
        return ml_model

    def update(self, id, log, data):
        """
        Функция либо переобучает модель, либо выдает ошибку, что такой модели ещё нет, надо создать новую
        :param id: integer: уникальный идентификатор модели
        :param data: json с новыми параметрами для модели
        :return: ничего не выдает
        """
        ml_model = None
        for model in self.ml_models['data']:
            if model['id'] == id:
                ml_model = model  # json со старыми параметрами
        if ml_model is None:
            log.error(f'%s failed to update, ml_model {id} does not exist')
            raise NotImplementedError('Такой модели ещё нет, надо создать новую')
        else:
            if (data['name'] == ml_model['name']) and (data['problem'] == ml_model['problem']):
                ml_model.update(data)  # кладу в них новые данные, 'X', 'y', 'h_tune'
                self.create(ml_model, log, is_new=False)  # переобучаю модель
                log.info(f'ml_model {ml_model["id"]} %s successfully updated')
            else:
                log.error(
                    f'%s failed to update - model with name {ml_model["name"]}, problem {ml_model["problem"]} and h_tune {ml_model["h_tune"]} does not exist yet')
                raise NotImplementedError('Такой модели ещё нет, надо создать новую')

    def delete(self, id):
        """
        Удаление модели по id
        :param id: integer: уникальный идентификатор модели
        :return: удаление модели из списка моделей
        """
        for model in self.ml_models['data']:
            if model['id'] == id:
                registered_model_name = f"{model['problem']}_{model['name']}_{model['id']}"
                all_versions = client.search_model_versions("name='{}'".format(registered_model_name))
                print(all_versions)
                for version in all_versions:
                    mlflow.delete_run(version.run_id)

                self.ml_models['data'].remove(model)

def classification(model, x, y, h_tune=False):
    """
    :param model: название класса для модели классификации (строка) - "Random Forest" или "SVM".
    :param x: np.array(): выборка с признаками для обучения.
    :param y: np.array(): таргеты.
    :param h_tune: boolean: нужен ли подбор гиперпараметров или нет.
    :return: model(): обученная модель.
    """
    if model == 'Random_Forest':
        param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 4], 'max_features': ['auto', 'sqrt']}
        clf = RandomForestClassifier(random_state=0)
    elif model == 'SVM':
        param_grid = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        clf = SVC(random_state=0)

    if h_tune:
        clf_cv = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
        clf_cv.fit(x, y)
        return clf_cv.best_estimator_
    else:
        clf.fit(x, y)
        return clf

def regression(model, x, y, h_tune=False):
    """
    :param model: название класса для модели регрессии (строка) - "Random Forest" или "SVM".
    :param x: np.array(): выборка с признаками для обучения.
    :param y: np.array(): таргеты.
    :param h_tune: boolean: нужен ли подбор гиперпараметров или нет.
    :return: model(): обученная модель.
    """
    if model == 'Random_Forest':
        param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 4], 'max_features': ['auto', 'sqrt']}
        lr = RandomForestRegressor(random_state=0)
    elif model == 'SVM':
        param_grid = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        lr = SVR()

    if h_tune:
        lr_cv = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5)
        lr_cv.fit(x, y)
        return lr_cv.best_estimator_

    else:
        lr.fit(x, y)
        return lr
