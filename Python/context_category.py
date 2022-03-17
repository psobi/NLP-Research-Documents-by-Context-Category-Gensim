#File: context_category.py
#Author: Pawel Sobieralski

import se_utils #Silenteight utils
import pandas as pd
import numpy as np
from numpy import linalg as la
import traceback
import sys

class ContextCategory():

    def __init__(self):
        pass

    def processDocument(self, corpus_doc):
        
        proc_doc = list(map(se_utils.process_document,corpus_doc)) 
        tokens_flat_list = [item for sublist in proc_doc for item in sublist]
        return tokens_flat_list


    def buildVocabulary(self, tc):

        vocab = list(set(tc))
        vocab.sort()
        
        vocab_dict = {}
        for index, word in enumerate(vocab):
            vocab_dict[word] = index
            
        return vocab_dict

    def buildContext(self, corpus_list, vocab_dict):

        co_occurrence_vectors = pd.DataFrame(
            np.zeros([len(vocab_dict), len(vocab_dict)]),
            index = vocab_dict.keys(),
            columns = vocab_dict.keys()
        )
        
        for index, element in enumerate(corpus_list):
            
            start = 0 if index-2 < 0 else index-2
            finish = len(corpus_list) if index+2 > len(corpus_list) else index+3
            
            context = corpus_list[start:index] + corpus_list[index+1:finish]
            for word in context:
                
                co_occurrence_vectors.loc[element, word] = (
                    co_occurrence_vectors.loc[element, word]+1
                )
                
        return co_occurrence_vectors/len(corpus_list) #Scale


    def cosineSimilarity(self, v1, v2):
        
        denominator = la.norm(v1) * la.norm(v2)
        
        if denominator > 0:
            return v1.dot(v2) / denominator
        else:
            return 0
    

    def compareByContext(self, doc1, doc2, dimensions):
        
        angle = 0.0
        for i, d in enumerate(dimensions):
            
            #Reverse cosine to radians to get avg
            angle += np.arccos(self.cosineSimilarity(np.array(doc1[d]), np.array(doc2[d])))
            angle = angle / len(dimensions)
            
            return (dimensions,np.cos(angle))

def main():

    try:

        print("Running Context Category")

        train_corpus = [
            "Human machine accusations for lab allegations applications",
            "A survey of user opinion of allegations and charges response time",
            "The EPS user accusations management charges",
            "Charges and allegations engineering testing of EPS",
            "Relation of user perceived response time to error measurement",
            "The generation of random binary unordered conviction",
            "The intersection sentencing of paths in conviction",
            "Sentencing minors IV Widths of conviction and well quasi ordering",
            "Sentencing minors A survey",
            ]


        #Context
        context1 = ['allegations','accusations','charges']
        context2 = ['conviction','sentencing']

        #Unseen before document
        test_document = ["A survey of user opinion of allegations and charges response time"]

        cc = ContextCategory()

        #Preprocessing
        train_doc = cc.processDocument(train_corpus)
        test_doc = cc.processDocument(test_document)

        #Build vector space
        vocab_dict = cc.buildVocabulary(train_doc)

        #Build context vectors
        train_vector = cc.buildContext(train_doc,vocab_dict)
        test_vector = cc.buildContext(test_doc,vocab_dict)

        #Compare documents in given category
        cc.compareByContext(train_vector, test_vector, context1)

        sys.exit(0)

    except Exception:
        print("Compare documents by context category failed")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
