import os
import json
from pathlib import Path
from datasets import load_from_disk
from named_entity_recognition import logger
from named_entity_recognition.entity.config_entity import ModelEvaluationConfig
from named_entity_recognition.transformer.model import NERModel
from named_entity_recognition.transformer.losses import MaskedLoss
from named_entity_recognition.utils.common import save_json
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class ModelEvaluation:

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self):
        
        test = load_from_disk(self.config.data_path)['test']
        test_input = tf.data.Dataset.from_tensor_slices(tf.concat((test['input_ids'],test['attention_mask']), axis = -1)).batch(self.config.params_batch_size)
        y_true = test['labels']
        y_pred = self.model.predict(test_input)
        loss = MaskedLoss()(y_true, y_pred)
        metrics = self.compute_metrics(y_true, y_pred)

        self.scores = {
            'loss': float(loss),
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        }
        save_json(Path(os.path.join(self.config.root_dir,'scores.json')), self.scores)

    def get_model(self):

        self.model = NERModel(
            num_tags=self.config.params_num_tags,
            num_layers=self.config.params_num_encoder_layers,
            embedding_dim=self.config.params_embedding_dim,
            fully_connected_dim=self.config.params_fully_connected_dim,
            num_heads=self.config.params_num_heads,
            vocab_size=self.config.params_vocab_size,
            max_positional_encoding=self.config.params_max_positional_encoding_length
        )

        self.model.load_weights(self.config.model_weights_path).expect_partial()
        

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
