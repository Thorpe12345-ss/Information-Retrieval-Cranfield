import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from math import log2

# Unchanged functions
def load_cranfield_docs(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    documents = []
    for doc in root.findall('doc'):
        docno = doc.find('docno').text.strip() if doc.find('docno') is not None and doc.find('docno').text is not None else "Unknown"
        title = doc.find('title').text.strip() if doc.find('title') is not None and doc.find('title').text is not None else "Unknown"
        author = doc.find('author').text.strip() if doc.find('author') is not None and doc.find('author').text is not None else "Unknown"
        bib = doc.find('bib').text.strip() if doc.find('bib') is not None and doc.find('bib').text is not None else "Unknown"
        text = doc.find('text').text.strip() if doc.find('text') is not None and doc.find('text').text is not None else "Unknown"
        documents.append({
            'docno': docno,
            'title': title,
            'author': author,
            'bib': bib,
            'text': text
        })
    return documents

def build_tfidf_matrix(documents):
    doc_texts = [doc['text'] for doc in documents]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(doc_texts)
    doc_ids = [doc['docno'] for doc in documents]
    return vectorizer, tfidf_matrix, doc_ids

def search(query, tfidf_matrix, vectorizer):
    query_vec = vectorizer.transform([query])
    print(f"Query Vector for '{query}':\n{query_vec.toarray()}")
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix)
    print(f"Cosine Similarities for query '{query}': {cosine_similarities}")
    ranked_indices = cosine_similarities.argsort()[0][::-1]
    return ranked_indices, cosine_similarities

def generate_output(ranked_indices, cosine_similarities, doc_ids):
    output_lines = []
    for rank, index in enumerate(ranked_indices):
        similarity = cosine_similarities[0][index]
        doc_id = doc_ids[index]
        output_lines.append(f"1 0 {doc_id} {rank + 1} {similarity} vsm_run")
    return output_lines

# Updated load_qrels with more debugging
def load_qrels(qrels_path):
    qrels = {}
    try:
        with open(qrels_path, 'r') as f:
            lines = f.readlines()
            print(f"First 5 lines of {qrels_path}: {lines[:5]}")  # Show file content
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 4:
                    print(f"Skipping malformed line: {line.strip()}")
                    continue
                query_id, _, doc_id, relevance = parts
                if int(relevance) > 0:
                    if query_id not in qrels:
                        qrels[query_id] = set()
                    qrels[query_id].add(doc_id)
        print("Loaded qrels:", {k: list(v) for k, v in qrels.items()})
    except FileNotFoundError:
        print(f"Error: {qrels_path} not found!")
    return qrels

# Unchanged metric functions
def precision_at_k(ranked_doc_ids, relevant_docs, k=5):
    top_k = ranked_doc_ids[:k]
    relevant_count = len([doc for doc in top_k if doc in relevant_docs])
    return relevant_count / k

def average_precision(ranked_doc_ids, relevant_docs):
    relevant_count = 0
    precision_sum = 0
    for rank, doc_id in enumerate(ranked_doc_ids, 1):
        if doc_id in relevant_docs:
            relevant_count += 1
            precision_sum += relevant_count / rank
    return precision_sum / len(relevant_docs) if relevant_docs else 0

def ndcg_at_k(ranked_doc_ids, relevant_docs, k=None):
    if not relevant_docs:
        return 0
    k = k or len(ranked_doc_ids)
    dcg = 0
    for i, doc_id in enumerate(ranked_doc_ids[:k], 1):
        if doc_id in relevant_docs:
            dcg += 1 / log2(i + 1)
    ideal_dcg = sum(1 / log2(i + 1) for i in range(1, min(len(relevant_docs), k) + 1))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0

if __name__ == "__main__":
    print("Running Vector Space Model...")
    cranfield_docs_path = "cran.all.1400.xml"
    documents = load_cranfield_docs(cranfield_docs_path)
    print("Loaded Documents ID: 1 - 1400")
    vectorizer, tfidf_matrix, doc_ids = build_tfidf_matrix(documents)
    print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

    output_file = "vsm_output.txt"
    with open(output_file, "w") as f:
        f.write("Query Number;Iteration;Document ID;Rank Position;Relevance Score;Model Run\n")
        print(f"Header written to {output_file}")

    # Test with known Cranfield queries (adjust after checking cran.qry.xml)
    queries = [
        "high-speed viscous flow past a two-dimensional body",
        "wing aerodynamics"   # Assume ID "2"
    ]
    query_to_id = {queries[0]: "1", queries[1]: "2"}

    qrels_path = "cranqrel.trec.txt"
    qrels = load_qrels(qrels_path)

    map_scores = []
    p5_scores = []
    ndcg_scores = []

    for query in queries:
        print(f"Results for query: {query}")
        ranked_indices, cosine_similarities = search(query, tfidf_matrix, vectorizer)
        output_lines = generate_output(ranked_indices, cosine_similarities, doc_ids)
        ranked_doc_ids = [doc_ids[idx] for idx in ranked_indices[:10]]

        print('/Query Number/Iteration/Related Document/Document ID/Rank Position/Relevance Score/Model Run')
        for line in output_lines[:5]:
            print(line)
        with open(output_file, "a") as f:
            for line in output_lines[:10]:
                f.write(line + "\n")

        query_id = query_to_id[query]
        relevant_docs = qrels.get(query_id, set())
        print(f"Ranked doc IDs: {ranked_doc_ids}")
        print(f"Relevant docs for Query ID {query_id}: {relevant_docs}")

        p5 = precision_at_k(ranked_doc_ids, relevant_docs, k=5)
        ap = average_precision(ranked_doc_ids, relevant_docs)
        ndcg = ndcg_at_k(ranked_doc_ids, relevant_docs)

        print(f"Metrics for '{query}' (Query ID {query_id}):")
        print(f"P@5: {p5:.4f}")
        print(f"AP: {ap:.4f}")
        print(f"NDCG: {ndcg:.4f}")
        print("-" * 50)

        map_scores.append(ap)
        p5_scores.append(p5)
        ndcg_scores.append(ndcg)

    map_avg = sum(map_scores) / len(map_scores)
    p5_avg = sum(p5_scores) / len(p5_scores)
    ndcg_avg = sum(ndcg_scores) / len(ndcg_scores)
    print(f"Average Metrics:")
    print(f"MAP: {map_avg:.4f}")
    print(f"P@5: {p5_avg:.4f}")
    print(f"NDCG: {ndcg_avg:.4f}")

    print(f"VSM results saved to {output_file}")