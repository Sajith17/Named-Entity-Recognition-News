import os
import pickle
from datasets import load_from_disk
from named_entity_recognition import logger
from named_entity_recognition.utils.common import get_size
import tensorflow as tf
from named_entity_recognition.entity.config_entity import TokenizerPreparationConfig

class TokenizerPreparation:

    def __init__(self, config: TokenizerPreparationConfig):
        self.config = config

    def prepare_tokenizer(self):
        data = load_from_disk(self.config.data_path)
        tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="[UNK]", lower=True)
        tokenizer.fit_on_texts([' '.join(x) for x in data['train']['tokens']])
        with open(os.path.join(self.config.root_dir, 'tokenizer.pickle'), 'wb') as f:
            pickle.dump(tokenizer, f)
        logger.info(f"Tokenizer created successfully at {self.config.root_dir}")