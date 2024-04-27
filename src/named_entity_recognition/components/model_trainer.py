import os
from pathlib import Path
import pickle
from datasets import load_from_disk
from named_entity_recognition import logger
from named_entity_recognition.entity.config_entity import ModelTrainerConfig
from named_entity_recognition.transformer.model import NERModel
from named_entity_recognition.transformer.optimizer import get_optimizer_with_custom_lr_sheduler
from named_entity_recognition.transformer.losses import MaskedLoss
from named_entity_recognition.transformer.metrics import masked_acc
import numpy as np
import tensorflow as tf

class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def get_model(self):

        model = NERModel(
            num_tags=self.config.params_num_tags,
            num_layers=self.config.params_num_encoder_layers,
            embedding_dim=self.config.params_embedding_dim,
            fully_connected_dim=self.config.params_fully_connected_dim,
            num_heads=self.config.params_num_heads,
            vocab_size=self.config.params_vocab_size,
            max_positional_encoding=self.config.params_max_positional_encoding_length
        )

        self.model = model


    def train(self):

        data = load_from_disk(str(self.config.data_path))

        optimizer = get_optimizer_with_custom_lr_sheduler(embedding_dim=self.config.params_embedding_dim)
        early_stopping = tf.keras.callbacks.EarlyStopping(
                                        monitor='val_loss',
                                        patience=3,
                                        restore_best_weights=True,
                                    )
        self.model.compile(optimizer=optimizer, loss=MaskedLoss(), metrics=[masked_acc])
        
        train = tf.data.Dataset.from_tensor_slices((data['train']['input_ids'],data['train']['attention_mask'],data['train']['labels'])).map(lambda x,y,z: (tf.concat((x,y),axis=-1),z)).shuffle(10000).batch(self.config.params_batch_size)
        val = tf.data.Dataset.from_tensor_slices((data['validation']['input_ids'],data['validation']['attention_mask'],data['validation']['labels'])).map(lambda x,y,z: (tf.concat((x,y),axis=-1),z)).batch(self.config.params_batch_size)

        self.model.fit(
            train,
            validation_data=(val),
            epochs=self.config.params_epochs,
            callbacks=[early_stopping]
        )
        

        self.save_model(os.path.join(self.config.root_dir,'model.keras'),self.model)
        


    @staticmethod
    def save_model(path: Path, model: tf.keras.models.Model):
        model.save(path)

    def load_tokenizer(self):
        with open(self.config.tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer

