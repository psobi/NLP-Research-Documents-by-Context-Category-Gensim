#File: text_classifier.py
#Author: Pawel Sobieralski

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np
import re
import string
import se_utils
import sys
import traceback

from abc import ABC, abstractmethod
 
class TextClassifier(ABC):
 
    text_corpus = []
    labels = []

    data = {}

    @abstractmethod
    def readInputData(self):
        pass
 
    @abstractmethod
    def vectorizeData(self, data, vocab_size):
        pass

    @abstractmethod
    def buildModel(self, vocab_size, embedding_dim, max_length):
        pass
    
    @abstractmethod
    def trainModel(self, model, data, num_epochs):
        pass

    @abstractmethod
    def validateModel(self, history):
        pass

    @abstractmethod
    def getVocabSize(self, raw_data):
        pass        

    @abstractmethod
    def validateModel(self, history):
        pass

    def standardizeData(self, input_data):

        return list(map(lambda row: str(row).lower(), input_data))

    def validationSplit(self, input_data, k = 0.8):

        split = round(len(input_data) * k )

        train_documents = input_data['document'][:split]
        train_label = input_data['label'][:split]

        test_documents = input_data['document'][split:]
        test_label = input_data['label'][split:]
    
        return {'train_documents' : np.array(train_documents),
                'train_labels'    : np.array(train_label),
                'test_documents'  : np.array(test_documents),
                'test_labels'     : np.array(test_label)
            }

class TFClasifier(TextClassifier):
    
    def __init__(self):
        pass

    def se_make_sample_corpus(self):
        self.text_corpus = [
            "Human machine accusations for lab allegations applications",
            "A survey of user opinion of allegations charges response time",
            "The EPS user accusations management charges",
            "charges and human charges engineering testing of EPS",
            "Relation of user perceived response time to error measurement",
            "The generation of random binary unordered conviction",
            "The intersection sentencing of paths in conviction",
            "Sentencing minors IV Widths of conviction and well quasi ordering",
            "Sentencing minors A survey",
        ]
        self.labels = [0,0,0,0,0,1,1,1,1]

        return {'document':self.text_corpus,'label':self.labels}
    
    def readInputData(self):
        return self.se_make_sample_corpus()

    def vectorizeData(self, data, vocab_size, max_length):
    
        tokenizer = Tokenizer(num_words=vocab_size, oov_token='oov_tok')
        tokenizer.fit_on_texts(data['train_documents'])

        sequences = tokenizer.texts_to_sequences(data['train_documents'])
        training_ready = pad_sequences(sequences, maxlen = max_length, truncating='post')

        testing_documents = tokenizer.texts_to_sequences(data['test_documents'])
        testing_ready = pad_sequences(testing_documents, maxlen=max_length)
        
        return {'train_documents' : np.array(training_ready),
                'train_labels'    : data['train_labels'],
                'test_documents'  : np.array(testing_ready),
                'test_labels'     : data['test_labels'],
            }

    def buildModel(self, vocab_size, embedding_dim, max_length):
        
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(6, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            
        return model

    def trainModel(self, model, data, num_epochs):
        
        history = model.fit(data['train_documents'],
                            data['train_labels'],
                            epochs=num_epochs,
                            validation_data=(data['test_documents'],
                                            data['test_labels']
                                            )
                        )
        return history

    def validateModel(self, history):

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs=range(len(acc))

        return (epochs, acc, val_acc, loss, val_loss)

    def getVocabSize(self, raw_data):
       
        proc_doc = list(map(se_utils.process_document,raw_data['document'])) 
        return len([item for sublist in proc_doc for item in sublist])


def main():

    try:
        
        print("Running Text Classifier")

        embedding_dim = 10  #Each word will be represented by a n-dimensional vector
        max_length = 10     # Each document padded to n words
        trunc_type = 'post' # Truncated at the end         
        oov_tok = '<OOV>'   # Unknown token        
        padding_type = 'post' # Padding

        num_epochs = 20

        classifier = TFClasifier()

        raw_data = classifier.readInputData()

        raw_data['document'] = classifier.standardizeData(raw_data['document'])

        data = classifier.validationSplit(raw_data)

        vocab_size = classifier.getVocabSize(raw_data)

        data_training_ready = classifier.vectorizeData(data,vocab_size,max_length = 10)

        model = classifier.buildModel(vocab_size, embedding_dim, max_length)

        history = classifier.trainModel(model, data_training_ready, num_epochs)

        classifier.validateModel(history)

        sys.exit(0)

    except Exception:

        print("Compare documents by context category failed")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
