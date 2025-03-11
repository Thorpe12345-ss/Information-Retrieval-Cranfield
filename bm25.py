import math
from collections import Counter
import xml.etree.ElementTree as ET
import os

docs_file_path = 'cran.all.1400.xml'
queries_file_path = 'cran.qry.xml'
output_file = 'bm25_output.txt'
qrels_file_path = 'cranqrel.trec.txt'

def load_cranfield_docs(file_path):
    if not os.path.exists(file_path):
        print(f"❌ {file_path} not found!")
        return []
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
    print(f"✅ Loaded {len(documents)} documents from {file_path}")
    return documents

def load_queries(file_path):
    if not os.path.exists(file_path):
        print(f"❌ {file_path} not found!")
        return []
    tree = ET.parse(file_path)
    root = tree.getroot()
    queries = []
    for query in root.findall('top'):
        num_elem = query.find('num')
        num = num_elem.text.strip().replace('Number:', '').strip() if num_elem is not None and num_elem.text is not None else "Unknown"
        title = query.find('title').text.strip() if query.find('title') is not None and query.find('title').text is not None else "Unknown"
        queries.append({'num': num, 'title': title})
    print(f"✅ Loaded {len(queries)} queries from {file_path}")
    return queries

def compute_doc_term_counts(documents):
    return [Counter(doc['text'].lower().split()) for doc in documents]

def bm25_score(query, doc_term_counts, doc_lengths, k1=1.5, b=0.75):
    query_terms = query.lower().split()
    scores = []
    avg_doc_length = sum(doc_lengths) / len(doc_lengths)
    idf = {}
    num_docs = len(doc_lengths)
    for doc in doc_term_counts:
        for term in doc:
            idf[term] = idf.get(term, 0) + 1
    idf = {term: math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1) for term, freq in idf.items()}
    for i, term_counts in enumerate(doc_term_counts):
        score = 0
        doc_length = doc_lengths[i]
        for term in query_terms:
            if term in term_counts:
                term_freq = term_counts[term]
                numerator = idf.get(term, 0) * term_freq * (k1 + 1)
                denominator = term_freq + k1 * (1 - b + b * (doc_length / avg_doc_length))
                score += numerator / denominator
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def write_output(query_id, results, run_id, doc_ids, output_file):
    with open(output_file, "a") as f:
        rank = 1
        print(f"Writing top 10 results for query {query_id} to {output_file}")
        for doc_idx, score in results[:10]:
            doc_id = doc_ids[doc_idx]
            line = f"{query_id} 0 {doc_id} {rank} {score} {run_id}"
            f.write(line + "\n")
            rank += 1

def load_qrels(qrels_path):
    qrels = {}
    try:
        with open(qrels_path, 'r') as f:
            lines = f.readlines()
            print(f"First 5 lines of {qrels_path}: {lines[:5]}")
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

def precision_at_k(ranked_doc_ids, relevant_docs, k=5):
    top_k = ranked_doc_ids[:k]
    relevant_count = len([doc for doc in top_k if doc in relevant_docs])
    print(f"P@5: {relevant_count} relevant in top {k}")
    return relevant_count / k

def average_precision(ranked_doc_ids, relevant_docs):
    relevant_count = 0
    precision_sum = 0
    for rank, doc_id in enumerate(ranked_doc_ids, 1):
        if doc_id in relevant_docs:
            relevant_count += 1
            precision_sum += relevant_count / rank
    ap = precision_sum / len(relevant_docs) if relevant_docs else 0
    print(f"AP: {relevant_count} relevant, sum = {precision_sum}, total relevant = {len(relevant_docs)}")
    return ap

def ndcg_at_k(ranked_doc_ids, relevant_docs, k=None):
    if not relevant_docs:
        return 0
    k = k or len(ranked_doc_ids)
    dcg = 0
    for i, doc_id in enumerate(ranked_doc_ids[:k], 1):
        if doc_id in relevant_docs:
            dcg += 1 / math.log2(i + 1)
    ideal_dcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant_docs), k) + 1))
    print(f"NDCG: DCG = {dcg}, Ideal DCG = {ideal_dcg}")
    return dcg / ideal_dcg if ideal_dcg > 0 else 0

if __name__ == "__main__":
    print("Running BM25 Model...")
    print(f"Current working directory: {os.getcwd()}")
    documents = load_cranfield_docs(docs_file_path)
    queries = load_queries(queries_file_path)
    if not documents or not queries:
        print("Error: Documents or queries failed to load. Exiting.")
        exit(1)
    doc_term_counts = compute_doc_term_counts(documents)
    doc_lengths = [sum(counts.values()) for counts in doc_term_counts]
    doc_ids = [doc['docno'] for doc in documents]

    output_file = "bm25_output.txt"
    with open(output_file, "w") as f:
        f.write("Query Number;Iteration;Document ID;Rank Position;Relevance Score;Model Run\n")
        print(f"Header written to {output_file}")
    print(f"Cleared {output_file}")

    selected_queries = [
        "high-speed viscous flow past a two-dimensional body",
        "wing aerodynamics"   
    ]
    query_to_id = {selected_queries[0]: "47", selected_queries[1]: "192"}  # Verify in cran.qry.xml

    qrels = load_qrels(qrels_file_path)
    if not qrels:
        print("Error: Failed to load qrels. Exiting.")
        exit(1)

    map_scores = []
    p5_scores = []
    ndcg_scores = []

    for query in selected_queries:
        query_id = query_to_id[query]
        print(f"Results for query: {query} (Query ID {query_id})")
        try:
            bm25_results = bm25_score(query, doc_term_counts, doc_lengths)
            write_output(query_id, bm25_results, "bm25_run", doc_ids, output_file)
            
            ranked_doc_ids = [doc_ids[doc_idx] for doc_idx, _ in bm25_results[:10]]
            print('/Query Number/Iteration/Related Document/Document ID/Rank Position/Relevance Score/Model Run')
            for rank, (doc_idx, score) in enumerate(bm25_results[:5], start=1):
                print(f"{query_id} 0 {doc_ids[doc_idx]} {rank} {score} bm25_run")

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
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            continue

    if map_scores:
        map_avg = sum(map_scores) / len(map_scores)
        p5_avg = sum(p5_scores) / len(p5_scores)
        ndcg_avg = sum(ndcg_scores) / len(ndcg_scores)
        print(f"Average Metrics:")
        print(f"MAP: {map_avg:.4f}")
        print(f"P@5: {p5_avg:.4f}")
        print(f"NDCG: {ndcg_avg:.4f}")
    else:
        print("No metrics computed due to errors.")

    print(f"BM25 results saved to {output_file}")