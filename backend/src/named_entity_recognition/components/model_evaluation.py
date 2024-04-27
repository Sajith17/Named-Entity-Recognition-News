import os
import json
from pathlib import Path
from datasets import load_from_disk
from named_entity_recognition import logger
from named_entity_recognition.entity.config_entity import ModelEvaluationConfig
from named_entity_recognition.transformer.model import NERModel
from named_entity_recognition.transformer.losses import MaskedLoss
from named_entity_recognition.transformer.metrics import masked_acc
from named_entity_recognition.transformer.optimizer import CustomSchedule
from named_entity_recognition.utils.common import save_json
import numpy as np
import tensorflow as tf
import mlflow
from urllib.parse import urlparse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class ModelEvaluation:

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self):
        
        test = load_from_disk(str(self.config.data_path))['test']
        test_input = tf.data.Dataset.from_tensor_slices(tf.concat((test['input_ids'],test['attention_mask']), axis = -1)).batch(self.config.params_batch_size)
        y_true = test['labels']
        y_pred = self.model.predict(test_input)
        loss = MaskedLoss()(y_true, y_pred)
        metrics = self.compute_metrics(y_true, y_pred)

        self.scores = {
            'loss': float(loss),
            **metrics,
        }
        save_json(Path(os.path.join(self.config.root_dir,'scores.json')), self.scores)

    def get_model(self):
        self.model = tf.keras.models.load_model(self.config.model_path, custom_objects={
            'NERModel':NERModel,
            'MaskedLoss':MaskedLoss,
            'masked_acc': masked_acc,
            'CustomSchedule': CustomSchedule
        })

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                self.scores
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model.keras", registered_model_name="NERModel")
            else:
                mlflow.keras.log_model(self.model, "model.keras")
        

    @staticmethod
    def compute_metrics(labels, pred_logits):

        pred_logits = np.argmax(pred_logits, axis=-1)

        predictions = []
        for prediction, label in zip(pred_logits, labels):
            for eval_pred, l in zip(prediction, label):
                if l!= -1:
                    predictions.append(eval_pred)

        true_labels = []
        for prediction, label in zip(pred_logits, labels):
            for eval_pred, l in zip(prediction, label):
                if l!= -1:
                    true_labels.append(l)

        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='macro')
        recall = recall_score(true_labels, predictions, average='macro')
        f1 = f1_score(true_labels, predictions, average='macro')
        
        return {'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1 }