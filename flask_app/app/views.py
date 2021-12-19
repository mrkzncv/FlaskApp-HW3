from flask_restx import Resource, fields
from app import api, models_dao, metrics, log

# шаблон с описанием сущности
# ml_models_desc = api.model('ML models', {'id': fields.Integer,
#                                          'problem': fields.String,
#                                          'name': fields.String,
#                                          # 'accuracy': fields.Float,
#                                          'h_tune': fields.Boolean,
#                                          'X': fields.List,  # ??
#                                          'y': fields.List,  # ??
#                                          'h_params': fields.List,
#                                          'prediction': fields.List})

implemented_models = {'classification': ['Random Forest', 'SVM'],
                      'regression': ['Random Forest', 'SVM']}


@api.route('/api/ml_models')
class MLModels(Resource):

    # Работает только с json
    @metrics.do_not_track()
    @metrics.counter('cnt_gets', 'Number of gets',
                     labels = {'status': lambda resp: resp.status_code})
    def get(self):
        """
        Возвращает доступные классы и информацию об обученных моделях.
        """
        return {'data': [implemented_models, models_dao.ml_models]}

    # @api.expect(ml_models_desc) # нужно проверить то, что отдает клиент, на валидность
    @metrics.counter('cnt_posts', 'Number of posts',
                     labels = {'status': lambda resp: resp.status_code})
    def post(self):
        """
        Обучение новой модели.
        """
        return models_dao.create(api.payload, log)

    def put(self):  # update?
        pass

    def delete(self):
        pass


@api.route('/api/ml_models/<int:id>')
class MLModelsID(Resource):

    @staticmethod
    def get(id):
    # log.info(f'id = {id}\n type(id) = {type(id)}')
        try:
            return models_dao.get(id, log)
        except NotImplementedError as e:
            api.abort(404, e)

    @staticmethod
    # @api.expect(ml_models_desc)
    def put(id):  # update
        """
        Переобучение модели на новых данных по id.
        """
        return models_dao.update(id, log, api.payload)

    @staticmethod
    # @api.expect(ml_models_desc)
    def delete(id):
        """
        Удаление данных о модели.
        """
        models_dao.delete(id)
        return '', 204

