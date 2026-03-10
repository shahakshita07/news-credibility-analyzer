import os
import joblib

class ModelLoader:
    """
    Singleton class to load model and vectorizer.
    """
    _model = None
    _vectorizer = None

    @staticmethod
    def get_model():
        if ModelLoader._model is None:
            model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
            if os.path.exists(model_path):
                ModelLoader._model = joblib.load(model_path)
        return ModelLoader._model

    @staticmethod
    def get_vectorizer():
        if ModelLoader._vectorizer is None:
            vec_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
            if os.path.exists(vec_path):
                ModelLoader._vectorizer = joblib.load(vec_path)
        return ModelLoader._vectorizer
