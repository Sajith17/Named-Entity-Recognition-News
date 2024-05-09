import os
import tensorflow as tf
import pickle
from named_entity_recognition import logger
from named_entity_recognition.transformer.model import NERModel
from named_entity_recognition.transformer.metrics import masked_acc
from named_entity_recognition.transformer.losses import MaskedLoss
from named_entity_recognition.transformer.optimizer import CustomSchedule
from named_entity_recognition.config.configuration import ConfigurationManager




class PredictionPipeline:
    def __init__(self):
        manager = ConfigurationManager()
        self.config = manager.get_data_transformation_config()
        self.model_path = manager.get_model_evaluation_config().model_path
        self.tags = {1:"PERSON", 3:"ORGANIZATION", 5: "LOCATION", 7: "MISCELLANEOUS"}
    def predict(self, sentence):
        input_ids, word_ids = self.get_input_ids_and_word_ids(sentence)
        input_ids = self.pad_and_mask([input_ids])
        labels = tf.argmax(self.model.predict(input_ids),axis=-1).numpy().tolist()[0]
        labels = self.align_labels(word_ids,labels)
        return {'entities': self.get_entities(sentence,labels)}

    def get_entities(self, sentence, labels):
        sentence = sentence.split(' ')
        entities = []
        current_span = ''
        prev_label = 0
        for word, label in zip(sentence, labels):
            if label == 0:
                if current_span:
                    entities.append([current_span, self.tags[prev_label]])
                    current_span = ''
                    prev_label = 0
                entities.append([word, None])
            elif label in {1,3,5,7}:
                if current_span:
                    entities.append([current_span, self.tags[prev_label]])
                current_span = word
                prev_label = label
            elif prev_label and label in {2,4,6,8}:
                current_span += ' ' + word
        if current_span:
            entities.append([current_span,self.tags[prev_label]])
        return entities
    
    # def get_entities(self, sentence, labels):
    #     sentence = sentence.split(' ')
    #     entities = []
    #     current_span = ''
    #     prev_label = 0
    #     for word, label in zip(sentence, labels):
    #         if label == 0 and current_span:
    #             entities.append((current_span, self.tags[prev_label]))
    #             current_span = ''
    #             prev_label = 0
    #         elif label in {1,3,5,7}:
    #             if current_span:
    #                 entities.append([current_span, self.tags[prev_label]])
    #             current_span = word
    #             prev_label = label
    #         elif prev_label and label in {2,4,6,8}:
    #             current_span += ' ' + word
    #     if current_span:
    #         entities.append((current_span, self.tags[prev_label]))
    #     return list(dict(entities).items())

    def align_labels(self, word_ids, labels):
        aligned_labels = [0]*(max(word_ids)+1)
        for word_id, label in zip(word_ids, labels):
            if aligned_labels[word_id] == 0 and label:
                aligned_labels[word_id] = label
        return aligned_labels
    
    def pad_and_mask(self, input_ids):
        input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=self.config.params_max_sequence_length, padding='post', truncating='post')
        mask = tf.cast(tf.math.not_equal(input_ids, 0), tf.int32)
        input = tf.concat((input_ids,mask),axis=-1)
        return input
        
    def get_input_ids_and_word_ids(self, sentence):
        tokenizer = self.get_tokenizer()
        tokens = tokenizer.texts_to_sequences(sentence.split(' '))
        input_ids = []
        word_ids = []
        
        for index, token in enumerate(tokens):
            for sub_token in token:
                if sub_token:
                    input_ids.append(sub_token)
                    word_ids.append(index)

        return input_ids, word_ids


    def get_tokenizer(self):
        with open(os.path.join('model','tokenizer.pickle'),'rb') as f:
            tokenizer = pickle.load(f) 
        return tokenizer
    
    def get_model(self):
        model = tf.keras.models.load_model(os.path.join('model','model.keras'), custom_objects={
            'NERModel':NERModel,
            'MaskedLoss':MaskedLoss,
            'masked_acc': masked_acc,
            'CustomSchedule': CustomSchedule
        })
        self.model =  model
    
if __name__ == '__main__':
    try:
        prediction_pipeline = PredictionPipeline()
        prediction_pipeline.get_model()
        entities = prediction_pipeline.predict("Ukraine launched drone attacks on Russia's Kushchevsk military airfield in the southern Krasnodar region, as well as two oil refineries, a source with knowledge of the operation told CNN.")
        print(entities)
    except Exception as e:
        logger.exception(e)
        raise e
        


