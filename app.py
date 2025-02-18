from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import joblib
from tensorflow.keras.preprocessing import image
import os
import logging
from functools import lru_cache

# Módulo recomendador
from recommender import ImprovedRecommender

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clase para desencriptar pickle del recomendador
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "ImprovedRecommender":
            module = "recommender"
        return super().find_class(module, name)

app = FastAPI(title="E-commerce ML Models API")

# Modelos de validación
class RecommenderInput(BaseModel):
    user_id: Optional[int] = None
    product_history: Optional[List[int]] = None
    n_recommendations: int = 5

class SalesPredictionInput(BaseModel):
    store: int
    dept: int
    date: str

# Manejador de modelos optimizado
class ModelManager:
    def __init__(self):
        self._recommender = None
        self._classifier = None
        self._sales_model = None
        self._scaler = None
        self._products_df = None
        self._interactions_df = None
        self._train_data = None
        self._stores_data = None
        self._features_data = None
        self._last_load = {}
        self._model_cache = {}
        self.load_initial_models()

    def load_initial_models(self):
        """Carga inicial de modelos y datos"""
        try:
            logger.info("Loading initial models and data...")
            self._load_recommender_system()
            self._load_classifier()
            self._load_sales_model()
            logger.info("Initial models loaded successfully")
        except Exception as e:
            logger.error(f"Error in initial model loading: {str(e)}")
            raise

    def _load_recommender_system(self):
        """Carga el sistema de recomendación y sus datos"""
        try:
            if os.path.exists('models/recommender/recommender.pkl'):
                with open('models/recommender/recommender.pkl', 'rb') as f:
                    self._recommender = CustomUnpickler(f).load()
                self._products_df = pd.read_pickle('models/recommender/products_df.pkl')
                self._interactions_df = pd.read_pickle('models/recommender/interactions_df.pkl')
                self._recommender.process_data(self._products_df, self._interactions_df)
                self._last_load['recommender'] = datetime.now()
                logger.info("Recommender system loaded successfully")
            else:
                logger.warning("Recommender model files not found")
        except Exception as e:
            logger.error(f"Error loading recommender system: {str(e)}")
            self._recommender = None
            raise

    def _load_classifier(self):
        """Carga el modelo de clasificación"""
        try:
            if os.path.exists('models/classifier/ecommerce_classifier.h5'):
                self._classifier = tf.keras.models.load_model(
                    'models/classifier/ecommerce_classifier.h5',
                    compile=False
                )
                self._classifier.make_predict_function()
                self._last_load['classifier'] = datetime.now()
                logger.info("Classifier loaded successfully")
            else:
                logger.warning("Classifier model file not found")
        except Exception as e:
            logger.error(f"Error loading classifier: {str(e)}")
            self._classifier = None
            raise

    def _load_sales_model(self):
        """Carga el modelo de predicción de ventas y datos relacionados"""
        try:
            if os.path.exists('models/sales/sales_prediction_model.h5'):
                self._sales_model = tf.keras.models.load_model(
                    'models/sales/sales_prediction_model.h5',
                    compile=False
                )
                self._scaler = joblib.load('models/sales/scaler.pkl')
                self._train_data = pd.read_csv('data/train.csv')
                self._stores_data = pd.read_csv('data/stores.csv')
                self._features_data = pd.read_csv('data/features.csv')
                self._last_load['sales_model'] = datetime.now()
                logger.info("Sales prediction model and data loaded successfully")
            else:
                logger.warning("Sales prediction model files not found")
        except Exception as e:
            logger.error(f"Error loading sales model: {str(e)}")
            self._sales_model = None
            raise

    def _check_reload_needed(self, model_name: str, reload_interval: int = 3600) -> bool:
        """Verifica si un modelo necesita ser recargado basado en el tiempo"""
        last_load = self._last_load.get(model_name)
        if last_load is None:
            return True
        return (datetime.now() - last_load).total_seconds() > reload_interval

    def clear_cache(self):
        """Limpia la caché de modelos"""
        self._recommender = None
        self._classifier = None
        self._sales_model = None
        self._last_load = {}
        self._model_cache = {}
        self.load_initial_models()

    @property
    def recommender(self):
        if self._recommender is None or self._check_reload_needed('recommender'):
            self._load_recommender_system()
        return self._recommender

    @property
    def classifier(self):
        if self._classifier is None or self._check_reload_needed('classifier'):
            self._load_classifier()
        return self._classifier

    @property
    def sales_model(self):
        if self._sales_model is None or self._check_reload_needed('sales_model'):
            self._load_sales_model()
        return self._sales_model

    @property
    def scaler(self):
        return self._scaler

    @property
    def train_data(self):
        return self._train_data

    @property
    def stores_data(self):
        return self._stores_data

    @property
    def features_data(self):
        return self._features_data

# Inicializar manejador de modelos (una sola vez)
models = ModelManager()

# Endpoints
@app.get("/")
def home():
    """Endpoint de salud y estado"""
    return {
        "status": "healthy",
        "models_available": [
            "recommender_system",
            "image_classifier",
            "sales_predictor"
        ],
        "server_time": datetime.now().isoformat()
    }

@app.post("/recommend")
def get_recommendations(input_data: RecommenderInput):
    """Endpoint para obtener recomendaciones de productos"""
    try:
        logger.info(f"Processing recommendation request: {input_data}")
        
        if models.recommender is None:
            raise HTTPException(
                status_code=503,
                detail="Recommender system is not available"
            )

        if input_data.user_id is not None:
            logger.info(f"Getting recommendations for user {input_data.user_id}")
            recommendations = models.recommender.get_recommendations(
                input_data.user_id,
                n_recommendations=input_data.n_recommendations
            )
        elif input_data.product_history is not None:
            logger.info(f"Getting recommendations from product history: {input_data.product_history}")
            recommendations = models.recommender.get_recommendations_from_history(
                input_data.product_history,
                n_recommendations=input_data.n_recommendations
            )
        else:
            logger.info("Getting popular recommendations")
            recommendations = models.recommender._get_popular_recommendations(
                n_recommendations=input_data.n_recommendations
            )

        return {
            "recommendations": recommendations,
            "metadata": {
                "request_type": "user_based" if input_data.user_id else 
                              "history_based" if input_data.product_history else 
                              "popular",
                "timestamp": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    """Endpoint para clasificar imágenes de productos"""
    try:
        logger.info(f"Processing image classification request for file: {file.filename}")
        
        if models.classifier is None:
            raise HTTPException(
                status_code=503,
                detail="Image classifier is not available"
            )

        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(
                status_code=400,
                detail="Only JPEG and PNG images are supported"
            )

        temp_path = f"temp_{file.filename}"
        try:
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            img = image.load_img(temp_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x/255.0

            prediction = models.classifier.predict(x, verbose=0)
            class_names = ['jeans', 'sofa', 'tshirt', 'tv']
            predicted_class = class_names[np.argmax(prediction)]
            confidence = float(np.max(prediction))

            return {
                "classification": {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "probabilities": {
                        class_name: float(prob) 
                        for class_name, prob in zip(class_names, prediction[0])
                    }
                },
                "metadata": {
                    "filename": file.filename,
                    "file_size": len(content),
                    "timestamp": datetime.now().isoformat()
                }
            }

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        logger.error(f"Error classifying image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-sales")
def predict_sales(input_data: SalesPredictionInput):
    """Endpoint para predecir ventas"""
    try:
        logger.info(f"Processing sales prediction request: {input_data}")
        
        if models.sales_model is None:
            raise HTTPException(
                status_code=503,
                detail="Sales prediction model is not available"
            )

        if input_data.store not in models.stores_data['Store'].values:
            raise HTTPException(status_code=400, detail=f"Invalid store: {input_data.store}")
        
        if input_data.dept not in models.train_data['Dept'].unique():
            raise HTTPException(status_code=400, detail=f"Invalid department: {input_data.dept}")
        
        try:
            date = pd.to_datetime(input_data.date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")
        
        if date not in models.features_data['Date'].values:
            raise HTTPException(
                status_code=400, 
                detail=f"No feature data available for date: {input_data.date}"
            )

        # Aquí iría la lógica de preparación de datos y predicción
        # Por ahora retornamos un placeholder
        return {
            "predictions": {
                "sales": 0.0  # Placeholder
            },
            "metadata": {
                "store": input_data.store,
                "department": input_data.dept,
                "date": date.strftime('%Y-%m-%d'),
                "timestamp": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Error predicting sales: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/status")
def get_models_status():
    """Endpoint para verificar el estado de los modelos"""
    try:
        return {
            "recommender": {
                "loaded": models._recommender is not None,
                "last_load": models._last_load.get('recommender')
            },
            "classifier": {
                "loaded": models._classifier is not None,
                "last_load": models._last_load.get('classifier')
            },
            "sales_model": {
                "loaded": models._sales_model is not None,
                "last_load": models._last_load.get('sales_model')
            },
            "data_loaded": {
                "train_data": models._train_data is not None,
                "stores_data": models._stores_data is not None,
                "features_data": models._features_data is not None
            },
            "server_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/check")
def check_system():
    """Verifica la existencia de archivos necesarios y estructura del sistema"""
    try:
        files_to_check = {
            'recommender': [
                'models/recommender/recommender.pkl',
                'models/recommender/products_df.pkl',
                'models/recommender/interactions_df.pkl'
            ],
            'classifier': [
                'models/classifier/ecommerce_classifier.h5'
            ],
            'sales': [
                'models/sales/sales_prediction_model.h5',
                'models/sales/scaler.pkl',
                'data/train.csv',
                'data/stores.csv',
                'data/features.csv'
            ]
        }

        status = {}
        for model, paths in files_to_check.items():
            status[model] = {
                'files_exist': all(os.path.exists(p) for p in paths),
                'files_status': {
                    p: {
                        'exists': os.path.exists(p),
                        'size': os.path.getsize(p) if os.path.exists(p) else 0
                    } for p in paths
                }
            }

        return {
            'system_check': status,
            'current_working_directory': os.getcwd(),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/reload")
def reload_models():
    """Endpoint para recargar todos los modelos"""
    try:
        models.clear_cache()
        return {"status": "success", "message": "All models reloaded successfully"}
    except Exception as e:
        logger.error(f"Error reloading models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommender/status")
def get_recommender_status():
    """Endpoint para verificar el estado del recomendador"""
    try:
        if models.recommender is None:
            return {
                "status": "not_loaded",
                "error": "Recommender system is not loaded",
                "timestamp": datetime.now().isoformat()
            }

        has_popular = hasattr(models.recommender, '_popular_recommendations')
        num_popular = len(models.recommender._popular_recommendations) if has_popular else 0
        
        return {
            "status": "loaded",
            "has_popular_recommendations": has_popular,
            "num_popular_recommendations": num_popular,
            "last_load": models._last_load.get('recommender'),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error checking recommender status: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
