import re
import string
import numpy as np

import numpy as np
from numpy import linalg as la

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

def process_document(doc, stem_flag=False):
   
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    # remove tickers like $GEdocTokenize
    doc = re.sub(r'\$\w*', '', doc)
    # remove RT"
    doc = re.sub(r'^RT[\s]+', '', doc)
    # hyperlinks    
    doc = re.sub(r'https?://[^\s\n\r]+', '', doc)
    # hashtags
    # only removing the hash # sign from the word
    doc = re.sub(r'#', '', doc)
    # tokenize docs
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    doc_tokens = tokenizer.tokenize(doc)

    docs_clean = []
    for word in doc_tokens:
        if (word not in stopwords_english and 
                word not in string.punctuation):  
            if stem_flag:
                word = stemmer.stem(word)
            docs_clean.append(word)

    return docs_clean
    

def se_cosine_similarity(v1,v2):
    
    denominator = la.norm(v1) * la.norm(v2)
    
    if denominator > 0:
        return v1.dot(v2) / denominator
    else:
        return 0