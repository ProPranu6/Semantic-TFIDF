#!/usr/bin/env python
# coding: utf-8

# In[1]:

import time
tic = time.time()
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from math import log
import gensim.downloader as gd
import numpy as np
from numpy.linalg import norm
import statistics as st



# In[25]:


class SemanticTFIDF:
    try:
        wvs
    except:
        word_vectors_name = 'fasttext-wiki-news-subwords-300' 
        print(f"Loading {word_vectors_name} word2vectors")
        wvs = gd.load(word_vectors_name)
    similarity_threshold = 0.45
    cosineMat = lambda A, B : np.dot(A,B)/(norm(A, axis=1)*norm(B))
    
    def __init__(self, docs):
        self.vocab_size = 0
        self.vocab = defaultdict(self.__up)
        self.docs = [self.tokenize(self.remove_stopwords_lemmatize(doc.lower())) for doc in docs]
        self.inverse_vocab = {v:k for k,v in self.vocab.items()}
        self.docs_freq_maps = [self.get_freq_map(doc) for doc in self.docs]
        self.N = len(docs)
        self.docwise_tfidf = [self.get_tfidf(doc_id) for doc_id in range(self.N)]
        self.vocab_vectors = self.get_word_vectors()
        
        
    def __up(self):
        self.vocab_size += 1
        return self.vocab_size
    
    @classmethod
    def reload_word_vectors(cls, new_word_vectors_name):
        del cls.wvs
        cls.word_vectors_name = new_word_vectors_name
        print(f"Loading {cls.word_vectors_name} word2vectors")
        cls.wvs = gd.load(cls.word_vectors_name)
        
    def tokenize(self, text):
        tokens = word_tokenize(text)
        for token in tokens:
            self.vocab[token] = self.vocab[token]
        return tokens
    
    def remove_stopwords_lemmatize(self, text):
        words = text.split()
        lemmatizer = WordNetLemmatizer()
        processed_text = " ".join([lemmatizer.lemmatize(word) for word in words if word not in stopwords.words("english")])
        return processed_text
            
    def get_freq_map(self, textlist):
        fmap = defaultdict(lambda : 0)
        
        for token in textlist:
            fmap[token] += 1
        return fmap
    
    def get_tfidf(self, doc_id):
        thisdoc_tfidf = defaultdict(lambda : 0)
        thisdoc_fmap = self.docs_freq_maps[doc_id]
        
        NOtherDocs = lambda x: sum([1 for dxid in range(self.N) if self.docs_freq_maps[dxid][x]>0])
        for word in thisdoc_fmap:
            thisdoc_tfidf[word] = thisdoc_fmap[word]*log(self.N/NOtherDocs(word))
            
        return thisdoc_tfidf
    
    def fetch_vector(self, word):
        if word in SemanticTFIDF.wvs:
            return SemanticTFIDF.wvs[word]
        elif word in self.vocab_vectors:
            return self.vocab_vectors[word]
        else:
            return np.random.random((300,))
        
    def get_word_vectors(self):
        vocab_vectors = np.empty((self.vocab_size, 300))
        for idd, token in self.inverse_vocab.items():
            vocab_vectors[idd-1] = SemanticTFIDF.wvs[token] if token in SemanticTFIDF.wvs else np.random.random((300,))
        return vocab_vectors
    
    def __sort_dict(self, d):
        max_final_doc_id = 1e+4
        max_final_doc_tfidf = 0
        for (k,v) in d.items():
            if v >= max_final_doc_tfidf:
                max_final_doc_tfidf = v
                max_final_doc_id = k
         
        return max_final_doc_id
    
    def __call__(self, query_string):
        #pointing_doc_ids = defaultdict(lambda : 0)
        
        query_string_doc_scores = np.zeros((self.N,))
        for query in self.tokenize(self.remove_stopwords_lemmatize(query_string.lower())):
            query_vector = self.fetch_vector(query.lower())

            arr = np.array([x if x>=SemanticTFIDF.similarity_threshold else 0 for x in SemanticTFIDF.cosineMat(self.vocab_vectors, query_vector)])
            if np.max(arr):
                word = self.inverse_vocab[np.argmax(arr)+1]

                max_tfidf = 0
                max_tfidf_doc_id = -1
                
                query_doc_scores = np.empty((self.N,))
                for dxid, doc_tfidf in enumerate(self.docwise_tfidf):
                    query_doc_scores[dxid] = doc_tfidf[word]
                    if doc_tfidf[word] >= max_tfidf:
                        max_tfidf_doc_id = dxid
                        max_tfidf = doc_tfidf[word] 

                if max_tfidf:
                    print(f"The given query word {query} best suits {word}") #in the document {max_tfidf_doc_id} - tfidf : {max_tfidf}")
                    query_string_doc_scores += query_doc_scores
                    #pointing_doc_ids[max_tfidf_doc_id] += max_tfidf
            
        if np.sum(query_string_doc_scores):
            finalised_doc_id = np.argmax(query_string_doc_scores) #self.__sort_dict(pointing_doc_ids)
            return " ".join(self.docs[finalised_doc_id]), finalised_doc_id
        else:
            return "None of the given words match well with the documents given, please query on relevant words", None
            


# In[19]:


SAMPLE_PARAGRAPHS = [
    "The COVID-19 pandemic has had a profound impact on the world, with millions of lives lost and economies severely impacted. From the initial outbreak in Wuhan, China, in late 2019, the virus quickly spread around the world, with governments struggling to keep up with the rapidly evolving situation. Lockdowns and other measures aimed at slowing the spread of the virus have had significant social and economic consequences, with many businesses closing and millions of people losing their jobs. Despite these challenges, vaccines have provided hope for a return to normalcy. However, vaccine distribution and uptake remain uneven, with some countries struggling to access vaccines while others have more than enough. The pandemic has underscored the importance of global cooperation and preparedness in the face of public health crises.",
    
    "Climate change is an urgent global challenge that has the potential to cause widespread environmental and social disruption. Rising temperatures, extreme weather events, and sea level rise are just some of the consequences of a warming planet. Efforts to reduce greenhouse gas emissions and transition to renewable energy sources are crucial to mitigate the worst effects of climate change. However, progress has been slow, with many countries continuing to rely on fossil fuels and failing to meet their emissions reduction targets. In addition to mitigation, adaptation measures are also needed to help communities and ecosystems cope with the impacts of climate change. The urgency of the climate crisis requires bold and concerted action from governments, businesses, and individuals alike.",
    
    "Artificial intelligence (AI) and automation are rapidly transforming many aspects of society, from transportation and healthcare to finance and entertainment. While these technologies have the potential to improve efficiency and enhance human well-being, they also raise ethical and social challenges. One concern is the potential for job displacement as AI and automation replace human labor. Another is the risk of bias and discrimination in decision-making algorithms. Ensuring that AI and automation are developed and deployed responsibly is crucial for realizing their full potential while minimizing their negative impacts. This requires collaboration between industry, government, academia, and civil society to develop ethical frameworks and guidelines for the development and deployment of these technologies.",
    
    "Social media platforms such as Facebook, Twitter, and Instagram have revolutionized communication and connection, but have also been criticized for enabling the spread of misinformation and contributing to polarization. The ease with which false information can be spread on these platforms has raised concerns about their impact on democracy and public discourse. In addition, algorithms used by social media platforms can create filter bubbles, reinforcing existing beliefs and limiting exposure to diverse perspectives. Efforts to combat misinformation and promote civil discourse online are necessary for a healthy democracy. This includes measures such as fact-checking and content moderation, as well as promoting media literacy and critical thinking skills.",
    
    "The ongoing struggle for racial and social justice has gained momentum in recent years, with protests and activism highlighting the need for systemic change. Issues such as police brutality, income inequality, and access to healthcare and education have come to the forefront of public consciousness. Addressing these issues requires not only policy changes, but also a deeper examination of the root causes of systemic racism and discrimination. This includes confronting the legacy of colonialism and slavery, and acknowledging the ways in which structures of power and privilege shape our society. Achieving true equality and justice will require sustained efforts from all sectors of society, including government, civil society, and the private sector."
]


tac = time.time()
print(f"The module load time : {tac-tic}s")

