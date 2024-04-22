import os
import pickle
from named_entity_recognition import logger
from named_entity_recognition.utils.common import get_size
import tensorflow as tf
from named_entity_recognition.entity.config_entity import TokenizerPreparationConfig

class TokenizerPreparation:

    def __init__(self, config: TokenizerPreparationConfig):
        self.config = config

    def prepare_tokenizer(self):

        if not os.path.exists(self.config.tokenizer_path):
            with open(self.config.data_path, 'rb') as f:
                data = pickle.load(f)
            tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="[UNK]", lower=True)
            tokenizer.fit_on_texts([' '.join(x) for x in data['train']['tokens']])
            with open(self.config.tokenizer_path, 'wb') as f:
                pickle.dump(tokenizer, f)
            logger.info(f"Tokenizer created successfully at {self.config.root_dir}")
        else: 
            logger.info(f"File already exists of size: {get_size(self.config.tokenizer_path)}")