import os
import pickle
from datasets import load_from_disk
from named_entity_recognition import logger
import numpy as np
import tensorflow as tf
from named_entity_recognition.entity.config_entity import DataTransformationConfig

class DataTransformation:

    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = self.load_tokenizer()

    def transform_data(self):
        save_path = os.path.join(self.config.root_dir,self.config.dataset_name)
        data = load_from_disk(str(self.config.data_path))
        data = data.map(self.tokenize_and_align_labels, batched=True).map(self.input_and_label_pad_sequence, batched=True).map(self.create_attention_mask, batched=True)
        data.save_to_disk(save_path)

    def load_tokenizer(self):
        with open(self.config.tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer
    
    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = [self.tokenizer.texts_to_sequences(token) for token in examples['tokens']]
        new_tokenized_inputs = []
        labels = []
        word_ids_list = []
        for i,tokenized_input in enumerate(tokenized_inputs):
            ner_tags = examples['ner_tags'][i]
            label_ids = []
            word_ids = []
            tokenized_sentence = []
            for j,tokenized_words in enumerate(tokenized_input):
                if tokenized_words:
                    tokenized_sentence.extend(tokenized_words)
                    word_ids.extend([j]*len(tokenized_words))
                    label_ids.append(ner_tags[j])
                    for k in range(len(tokenized_words)-1):
                        label_ids.append(ner_tags[j] if self.config.params_label_all_tokens else 0)
            labels.append(label_ids)
            word_ids_list.append(word_ids)
            new_tokenized_inputs.append(tokenized_sentence)
        return {'input_ids': new_tokenized_inputs, 'word_ids': word_ids_list, 'labels': labels}
    
    def input_and_label_pad_sequence(self, examples):
        return { 'input_ids': tf.keras.preprocessing.sequence.pad_sequences(examples['input_ids'],
                                                                            maxlen = self.config.params_max_sequence_length, 
                                                                            padding='post', truncating='post'),
                'labels': tf.keras.preprocessing.sequence.pad_sequences(examples['labels'],
                                                                        maxlen = self.config.params_max_sequence_length, 
                                                                        padding='post', truncating='post', value=-1 )}
    def create_attention_mask(self,examples):
        mask = 1 - (np.array(examples['input_ids'])==0)
        return {"attention_mask": mask}