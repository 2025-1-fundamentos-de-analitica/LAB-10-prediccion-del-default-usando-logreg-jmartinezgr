# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import pandas as pd
import numpy as np
import os
import gzip
import pickle
import json
import logging
import zipfile
from pathlib import Path
from typing import Dict, Tuple, Any, List
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuración centralizada del proyecto"""

    INPUT_PATH: str = "files/input/"
    MODELS_PATH: str = "files/models/"
    OUTPUT_PATH: str = "files/output/"
    TRAIN_FILE: str = "train_data.csv.zip"
    TEST_FILE: str = "test_data.csv.zip"
    TRAIN_CSV: str = "train_default_of_credit_card_clients.csv"
    TEST_CSV: str = "test_default_of_credit_card_clients.csv"
    MODEL_FILE: str = "model.pkl.gz"
    METRICS_FILE: str = "metrics.json"
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 10
    SCORING: str = "balanced_accuracy"
    CATEGORICAL_FEATURES: List[str] = None
    NUMERICAL_FEATURES: List[str] = None

    def __post_init__(self):
        if self.CATEGORICAL_FEATURES is None:
            self.CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]

        if self.NUMERICAL_FEATURES is None:
            self.NUMERICAL_FEATURES = [
                "LIMIT_BAL",
                "AGE",
                "PAY_0",
                "PAY_2",
                "PAY_3",
                "PAY_4",
                "PAY_5",
                "PAY_6",
                "BILL_AMT1",
                "BILL_AMT2",
                "BILL_AMT3",
                "BILL_AMT4",
                "BILL_AMT5",
                "BILL_AMT6",
                "PAY_AMT1",
                "PAY_AMT2",
                "PAY_AMT3",
                "PAY_AMT4",
                "PAY_AMT5",
                "PAY_AMT6",
            ]


class DataLoader:
    """Clase para cargar y procesar datos desde archivos ZIP"""

    def __init__(self, config: Config):
        self.config = config

    def load_data_from_zip(self, zip_path: str, csv_name: str) -> pd.DataFrame:
        """
        Carga datos desde un archivo CSV dentro de un ZIP

        Args:
            zip_path: Ruta al archivo ZIP
            csv_name: Nombre del archivo CSV dentro del ZIP

        Returns:
            DataFrame con los datos cargados

        Raises:
            FileNotFoundError: Si el archivo no existe
            Exception: Para otros errores de carga
        """
        try:
            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"El archivo {zip_path} no existe")

            logger.info(f"Cargando datos desde {zip_path}")

            with zipfile.ZipFile(zip_path, "r") as zip_file:
                with zip_file.open(csv_name) as csv_file:
                    df = pd.read_csv(csv_file)

            logger.info(f"Datos cargados exitosamente. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error al cargar datos desde {zip_path}: {str(e)}")
            raise

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y preprocesa el dataset siguiendo las especificaciones

        Args:
            df: DataFrame a limpiar

        Returns:
            DataFrame limpio
        """
        logger.info("Iniciando limpieza del dataset")

        # Crear copia para evitar modificar el original
        df_clean = df.copy()

        # Paso 1: Renombrar columna y eliminar ID
        df_clean = df_clean.rename(columns={"default payment next month": "default"})
        df_clean = df_clean.drop("ID", axis=1)

        # Eliminar registros con información no disponible
        df_clean = df_clean.dropna()

        # Filtrar valores inválidos en EDUCATION y MARRIAGE
        df_clean = df_clean[(df_clean["EDUCATION"] != 0) & (df_clean["MARRIAGE"] != 0)]

        # Agrupar valores de EDUCATION > 4 en categoría "others" (valor 4)
        df_clean.loc[df_clean["EDUCATION"] > 4, "EDUCATION"] = 4

        logger.info(f"Dataset limpio. Shape: {df_clean.shape}")
        return df_clean

    def split_features_target(
        self, df: pd.DataFrame, target_col: str = "default"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separa características y variable objetivo

        Args:
            df: DataFrame completo
            target_col: Nombre de la columna objetivo

        Returns:
            Tupla con (características, objetivo)
        """
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        return X, y


class ModelBuilder:
    """Clase para construir y entrenar modelos de regresión logística"""

    def __init__(self, config: Config):
        self.config = config

    def create_pipeline(self) -> Pipeline:
        """
        Crea pipeline de procesamiento y clasificación según especificaciones

        Pipeline incluye:
        - One-hot encoding para variables categóricas
        - MinMax scaling para variables numéricas (escala [0,1])
        - Selección de K mejores características
        - Regresión logística

        Returns:
            Pipeline configurado
        """
        logger.info("Creando pipeline de procesamiento")

        # Preprocessor con transformaciones específicas
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    self.config.CATEGORICAL_FEATURES,
                ),
                ("num", MinMaxScaler(), self.config.NUMERICAL_FEATURES),
            ],
            remainder="passthrough",
        )

        # Pipeline completo según especificaciones
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("selectkbest", SelectKBest(score_func=f_classif)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        solver="saga",
                        random_state=self.config.RANDOM_STATE,
                    ),
                ),
            ]
        )

        return pipeline

    def create_grid_search(self, pipeline: Pipeline) -> GridSearchCV:
        """
        Crea GridSearchCV para optimización de hiperparámetros

        Args:
            pipeline: Pipeline base

        Returns:
            GridSearchCV configurado
        """
        logger.info("Configurando GridSearchCV para optimización")

        param_grid = {
            "selectkbest__k": range(1, 11),  # Selección de 1 a 10 características
            "classifier__penalty": ["l1", "l2"],  # Regularización L1 y L2
            "classifier__C": [
                0.001,
                0.01,
                0.1,
                1,
                10,
                100,
            ],  # Parámetro de regularización
        }

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=self.config.CV_FOLDS,
            scoring=self.config.SCORING,
            refit=True,
            verbose=0,
            return_train_score=False,
            n_jobs=-1,  # Usar todos los cores disponibles
        )

        return grid_search

    def save_model(self, model: GridSearchCV, file_path: str) -> None:
        """
        Guarda modelo entrenado comprimido con gzip

        Args:
            model: Modelo entrenado
            file_path: Ruta donde guardar el modelo
        """
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            logger.info(f"Guardando modelo en {file_path}")
            with gzip.open(file_path, "wb") as f:
                pickle.dump(model, f)

            logger.info("Modelo guardado exitosamente")

        except Exception as e:
            logger.error(f"Error al guardar modelo: {str(e)}")
            raise


class MetricsCalculator:
    """Clase para calcular métricas de evaluación"""

    def calculate_classification_metrics(
        self, dataset_name: str, y_true: pd.Series, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calcula métricas de clasificación

        Args:
            dataset_name: Nombre del dataset ('train' o 'test')
            y_true: Valores reales
            y_pred: Valores predichos

        Returns:
            Diccionario con métricas
        """
        return {
            "type": "metrics",
            "dataset": dataset_name,
            "precision": precision_score(y_true, y_pred, average="binary"),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred, average="binary"),
            "f1_score": f1_score(y_true, y_pred, average="binary"),
        }

    def calculate_confusion_matrix_metrics(
        self, dataset_name: str, y_true: pd.Series, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calcula matriz de confusión

        Args:
            dataset_name: Nombre del dataset ('train' o 'test')
            y_true: Valores reales
            y_pred: Valores predichos

        Returns:
            Diccionario con matriz de confusión
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        return {
            "type": "cm_matrix",
            "dataset": dataset_name,
            "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
            "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
        }

    def save_metrics(self, metrics_list: List[Dict[str, Any]], file_path: str) -> None:
        """
        Guarda métricas en archivo JSON

        Args:
            metrics_list: Lista de métricas
            file_path: Ruta del archivo
        """
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            logger.info(f"Guardando métricas en {file_path}")
            with open(file_path, "w") as f:
                for metric in metrics_list:
                    json_line = json.dumps(metric)
                    f.write(json_line + "\n")

            logger.info("Métricas guardadas exitosamente")

        except Exception as e:
            logger.error(f"Error al guardar métricas: {str(e)}")
            raise


class LogisticCreditClassifier:
    """Clase principal que orquesta todo el proceso de clasificación logística"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.data_loader = DataLoader(self.config)
        self.model_builder = ModelBuilder(self.config)
        self.metrics_calculator = MetricsCalculator()

    def run(self) -> None:
        """Ejecuta todo el pipeline de entrenamiento y evaluación"""
        try:
            logger.info("Iniciando proceso de clasificación logística de crédito")

            # Paso 1: Cargar datos desde archivos ZIP
            train_df = self.data_loader.load_data_from_zip(
                os.path.join(self.config.INPUT_PATH, self.config.TRAIN_FILE),
                self.config.TRAIN_CSV,
            )
            test_df = self.data_loader.load_data_from_zip(
                os.path.join(self.config.INPUT_PATH, self.config.TEST_FILE),
                self.config.TEST_CSV,
            )

            # Paso 1: Limpiar datos
            train_df_clean = self.data_loader.clean_dataset(train_df)
            test_df_clean = self.data_loader.clean_dataset(test_df)

            # Paso 2: Dividir datasets en X, y
            X_train, y_train = self.data_loader.split_features_target(train_df_clean)
            X_test, y_test = self.data_loader.split_features_target(test_df_clean)

            logger.info(f"Datos de entrenamiento: {X_train.shape}")
            logger.info(f"Datos de prueba: {X_test.shape}")

            # Paso 3: Crear pipeline
            pipeline = self.model_builder.create_pipeline()

            # Paso 4: Optimizar hiperparámetros con validación cruzada
            grid_search = self.model_builder.create_grid_search(pipeline)

            logger.info("Iniciando entrenamiento y optimización de hiperparámetros...")
            grid_search.fit(X_train, y_train)
            logger.info("Entrenamiento completado")

            # Mostrar mejores parámetros
            logger.info(
                f"Mejor score de validación cruzada: {grid_search.best_score_:.4f}"
            )
            logger.info(f"Mejores parámetros: {grid_search.best_params_}")

            # Paso 5: Guardar modelo
            model_path = os.path.join(self.config.MODELS_PATH, self.config.MODEL_FILE)
            self.model_builder.save_model(grid_search, model_path)

            # Paso 6: Realizar predicciones
            logger.info("Realizando predicciones...")
            y_train_pred = grid_search.predict(X_train)
            y_test_pred = grid_search.predict(X_test)

            # Paso 6: Calcular métricas de clasificación
            train_metrics = self.metrics_calculator.calculate_classification_metrics(
                "train", y_train, y_train_pred
            )
            test_metrics = self.metrics_calculator.calculate_classification_metrics(
                "test", y_test, y_test_pred
            )

            # Paso 7: Calcular matrices de confusión
            train_cm = self.metrics_calculator.calculate_confusion_matrix_metrics(
                "train", y_train, y_train_pred
            )
            test_cm = self.metrics_calculator.calculate_confusion_matrix_metrics(
                "test", y_test, y_test_pred
            )

            # Guardar todas las métricas
            all_metrics = [train_metrics, test_metrics, train_cm, test_cm]
            metrics_path = os.path.join(
                self.config.OUTPUT_PATH, self.config.METRICS_FILE
            )
            self.metrics_calculator.save_metrics(all_metrics, metrics_path)

            # Mostrar resumen de resultados
            logger.info("Proceso completado exitosamente")
            logger.info(
                f"Precisión balanceada - Train: {train_metrics['balanced_accuracy']:.4f}"
            )
            logger.info(
                f"Precisión balanceada - Test: {test_metrics['balanced_accuracy']:.4f}"
            )
            logger.info(f"F1-Score - Train: {train_metrics['f1_score']:.4f}")
            logger.info(f"F1-Score - Test: {test_metrics['f1_score']:.4f}")

        except Exception as e:
            logger.error(f"Error en el proceso principal: {str(e)}")
            raise


def main():
    """Función principal"""
    try:
        # Crear instancia del clasificador con configuración por defecto
        classifier = LogisticCreditClassifier()

        # Ejecutar el proceso completo
        classifier.run()

    except Exception as e:
        logger.error(f"Error fatal: {str(e)}")
        raise


if __name__ == "__main__":
    main()
