import os
import pickle
import urllib.request as request
import zipfile
from pathlib import Path
from named_entity_recognition import logger
from named_entity_recognition.utils.common import get_size
import tensorflow as tf
import numpy as np
from named_entity_recognition.entity.config_entity import EmbeddingPreparationConfig

class EmbeddingPreparation:

    def __init__(self, config: EmbeddingPreparationConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_embedding_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_embedding_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_embedding_file))}")  
    
    def unzip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_embedding_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

    def prepare_word_to_vec(self):
        word_to_vec = {}
        with open(os.path.join(self.config.unzip_dir, f'glove.6B.{self.config.params_embedding_dims}d.txt'),'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:],'float32')
                word_to_vec[word]=vector
        self.word_to_vec = word_to_vec

    def prepare_tokenizer(self):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')
        tokenizer.fit_on_texts(self.word_to_vec.keys())
        self.tokenizer = tokenizer
        with open(os.path.join(self.config.root_dir, 'tokenizer.pickle'), 'wb') as f:
            pickle.dump(self.tokenizer, f)
    
    def prepare_embedding_weights(self):
        embedding_weights = np.zeros((self.config.params_vocab_size, self.config.params_embedding_dims))
        for word, i in self.tokenizer.word_index.items():
            if word in self.word_to_vec:
                embedding_weights[i] = self.word_to_vec[word]
        with open(os.path.join(self.config.root_dir, 'embedding.pickle'), 'wb') as f:
            pickle.dump(embedding_weights, f)
        

    

    
                