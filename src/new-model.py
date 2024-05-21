import pandas as pd
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import torch
import faiss
import os
import json
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# Set the environment variable to avoid OpenMP runtime errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the dataset files
train_file_path = 'WikiPassageQA/train.txt'
test_file_path = 'WikiPassageQA/test.txt'

# Read the files into pandas DataFrames
train_df = pd.read_csv(train_file_path, delimiter='\t')
test_df = pd.read_csv(test_file_path, delimiter='\t')

# Load the full document segments
with open('WikiPassageQA/document_passages.json', 'r') as file:
    full_documents = json.load(file)

def get_passage_text(document_id, segments):
    """Retrieve the passage text given a document ID and segments."""
    try:
        # Convert document ID to string
        document_id_str = str(document_id)
        
        # Check if document ID exists
        if document_id_str not in full_documents:
            raise KeyError(f"Document ID {document_id_str} not found")
        
        # Convert segments to a list of strings
        segment_list = segments.split(',')
        passage_texts = []

        # Retrieve and concatenate the text passages
        for segment in segment_list:
            segment_stripped = segment.strip()
            if segment_stripped not in full_documents[document_id_str]:
                raise KeyError(f"Segment {segment_stripped} not found in Document ID {document_id_str}")
            passage_texts.append((document_id_str, segment_stripped, full_documents[document_id_str][segment_stripped]))

        return passage_texts
    except KeyError as e:
        print(f"Error retrieving passage text: {e}")
        return f"Passage not found: {e}"

# Combine train and test data for full document encoding
combined_df = pd.concat([train_df, test_df])

# Encode all documents
all_passages_data = [get_passage_text(row['DocumentID'], row['RelevantPassages']) for _, row in combined_df.iterrows()]
all_passages = [' '.join(passage[2] for passage in passages) for passages in all_passages_data]

# Load the DPR context encoder and tokenizer
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

def encode_passages(passages, tokenizer, model, max_length=512, batch_size=16):
    embeddings = []
    for i in tqdm(range(0, len(passages), batch_size), desc="Encoding passages"):
        batch = passages[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        with torch.no_grad():
            embedding = model(**inputs).pooler_output
        embeddings.append(embedding)
    return torch.cat(embeddings)

# Encode passages if FAISS index does not exist
index_file = 'passage_index_ivf.faiss'
if not os.path.exists(index_file):
    print('Encoding passages...')
    passage_embeddings = encode_passages(all_passages, context_tokenizer, context_encoder)
    print('Done encoding passages!')

    # Create an IVF index
    nlist = 100  # Number of clusters
    quantizer = faiss.IndexFlatIP(passage_embeddings.shape[1])
    index = faiss.IndexIVFFlat(quantizer, passage_embeddings.shape[1], nlist)
    
    index.train(passage_embeddings.numpy())
    index.add(passage_embeddings.numpy())

    # Save the index
    faiss.write_index(index, index_file)
else:
    # Load the index
    print(f'Loading FAISS index from {index_file}...')
    index = faiss.read_index(index_file)
    print('FAISS index loaded!')

# Load the DPR question encoder and tokenizer
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

def encode_questions(questions, tokenizer, model, max_length=512, batch_size=16):
    embeddings = []
    for i in tqdm(range(0, len(questions), batch_size), desc="Encoding questions"):
        batch = questions[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        with torch.no_grad():
            embedding = model(**inputs).pooler_output
        embeddings.append(embedding)
    return torch.cat(embeddings)

def retrieve_passages(query, question_tokenizer, question_encoder, index, top_k=1):
    query_embedding = encode_questions([query], question_tokenizer, question_encoder)
    distances, indices = index.search(query_embedding.numpy(), top_k)
    return distances, indices

# Implement Hybrid Retrieval with BM25 and DPR
bm25_corpus = [text.split() for text in all_passages]
bm25 = BM25Okapi(bm25_corpus)

def hybrid_retrieve(query, question_tokenizer, question_encoder, index, bm25, top_k=1):
    # BM25 retrieval
    bm25_results = bm25.get_top_n(query.split(), all_passages, n=top_k*10)
    
    # DPR re-ranking
    dpr_scores = []
    for result in bm25_results:
        query_embedding = encode_questions([query], question_tokenizer, question_encoder)
        passage_embedding = encode_passages([result], context_tokenizer, context_encoder)
        dpr_score = torch.matmul(query_embedding, passage_embedding.T).item()
        dpr_scores.append(dpr_score)
    
    combined_results = sorted(zip(bm25_results, dpr_scores), key=lambda x: x[1], reverse=True)
    top_results = combined_results[:top_k]
    
    return top_results

# Retrieve and print the result for the first query from the test file
first_query = train_df['Question'].iloc[0]

# Retrieve the top-1 most relevant passage using hybrid retrieval
top_results = hybrid_retrieve(first_query, question_tokenizer, question_encoder, index, bm25, top_k=1)
top_passage = top_results[0][0]

# Print top_results for debugging
print("Top Results:", top_results)

# Find the document and segment IDs for the top passage
top_doc_id = None
top_seg_id = None
for passages in all_passages_data:
    for doc_id, segments, text in passages:
        if top_passage == text:  # Ensure exact match
            top_doc_id, top_seg_id = doc_id, segments
            break
    if top_doc_id is not None:
        break

# Print the result
print(f"Query: {first_query}")
print(f"Top-1 Retrieved Passage:")
print(f"Document ID: {top_doc_id}, Segment: {top_seg_id}")
print(f"Passage: {top_passage}")
''' Ca marche plus ...'''