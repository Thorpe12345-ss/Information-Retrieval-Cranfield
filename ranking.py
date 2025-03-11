import numpy as np
import math
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure stopwords is downloaded (only need to do this once)
nltk.download('punkt')
nltk.download('stopwords')

# Function to calculate term frequency (TF) for a term in a document
def compute_tf(document, term):
    return document.count(term) / len(document)

# Function to calculate inverse document frequency (IDF) for a term in all documents
def compute_idf(documents, term):
    doc_count = sum(1 for doc in documents if term in doc)
    return math.log(len(documents) / (1 + doc_count))

# Function to compute TF-IDF for a term in a document
def compute_tfidf(documents, document, term):
    tf = compute_tf(document, term)
    idf = compute_idf(documents, term)
    return tf * idf

# Vector Space Model for ranking documents
def vector_space_model(query, documents, inverted_index):
    query_tokens = word_tokenize(query.lower())
    query_tokens = [word for word in query_tokens if word not in stopwords.words('english')]  # Removing stopwords
    
    # Create the query vector
    query_vector = np.zeros(len(inverted_index))
    for i, term in enumerate(inverted_index.keys()):
        if term in query_tokens:
            query_vector[i] = 1  # Simple binary representation for query
    
    # Create document vectors based on the TF-IDF scores
    doc_vectors = []
    for doc_id in documents:
        doc_vector = np.zeros(len(inverted_index))
        for i, term in enumerate(inverted_index.keys()):
            doc_vector[i] = compute_tfidf(documents, documents[doc_id], term)
       
# Function to calculate the BM25 score for a document and query
def compute_bm25(query, documents, inverted_index, k1=1.5, b=0.75):
    avg_doc_len = np.mean([len(doc) for doc in documents])
    scores = {}
    
    for doc_id, doc in documents.items():
        score = 0
        doc_len = len(doc)
        
        for term in query:
            if term in inverted_index:
                # TF (Term Frequency) in the document
                tf = doc.count(term)
                # IDF (Inverse Document Frequency)
                idf = math.log(1 + (len(documents) - len(inverted_index[term]) + 0.5) / (len(inverted_index[term]) + 0.5))
                
                # BM25 formula
                score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len)))
        
        scores[doc_id] = score
    
    return scores

# Function to compute the language model (Unigram model)
def compute_language_model(query, documents, inverted_index, smoothing=0.1):
    scores = {}
    
    for doc_id, doc in documents.items():
        score = 0
        doc_len = len(doc)
        
        # Calculate term probabilities (with smoothing)
        for term in query:
            term_count = doc.count(term)
            prob = (term_count + smoothing) / (doc_len + smoothing * len(doc))
            score += math.log(prob)
        
        scores[doc_id] = score
    
    return scores
