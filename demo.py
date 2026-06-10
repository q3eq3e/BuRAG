# ============================================================
# RAG PROJECT
# ============================================================

# ============================================================
# INSTALL
# ============================================================

# !pip -q install llama-index
# !pip -q install llama-index-vector-stores-chroma
# !pip -q install llama-index-embeddings-huggingface
# !pip -q install chromadb
# !pip -q install sentence-transformers
# !pip -q install transformers
# !pip -q install accelerate
# !pip -q install bitsandbytes
# !pip -q install datasets
# !pip -q install rank-bm25
# !pip -q install spacy
# !python -m spacy download en_core_web_sm

# ============================================================
# IMPORTS
# ============================================================

import os
import gc
import torch
import chromadb
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from datasets import load_dataset

from rank_bm25 import BM25Okapi

from sentence_transformers import SentenceTransformer, CrossEncoder

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
    BitsAndBytesConfig,
)

from llama_index.core import VectorStoreIndex, Document, Settings

from llama_index.core.node_parser import SentenceSplitter

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.core.storage.storage_context import StorageContext
import re
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    # -----------------------------------------
    # DATA
    # -----------------------------------------
    "dataset_name": "rajpurkar/squad",
    "n_documents": 500,
    # -----------------------------------------
    # CHUNKING
    # -----------------------------------------
    "chunk_size": 256,
    "chunk_overlap": 50,
    "len_short_chunk_to_remove": 40,
    "len_short_article_to_remove": 200,
    # -----------------------------------------
    # EMBEDDINGS
    # -----------------------------------------
    "embedding_model": "BAAI/bge-small-en-v1.5",
    # -----------------------------------------
    # RERANKER
    # -----------------------------------------
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    # -----------------------------------------
    # GENERATOR
    # -----------------------------------------
    "test_mode": True,
    "test_generator": "sshleifer/tiny-gpt2",
    # "production_generator": "meta-llama/Meta-Llama-3-8B-Instruct",
    "production_generator": "mistralai/Mistral-7B-Instruct-v0.2",
    # -----------------------------------------
    # RETRIEVAL
    # -----------------------------------------
    "dense_top_k": 1,
    "bm25_top_k": 1,
    # -----------------------------------------
    # RERANKING
    # -----------------------------------------
    "rerank_top_k": 1,
    # -----------------------------------------
    # OTHER
    # -----------------------------------------
    "seed": 42,
}

# ============================================================
# MODEL SELECTION
# ============================================================

GENERATOR_MODEL = (
    CONFIG["test_generator"] if CONFIG["test_mode"] else CONFIG["production_generator"]
)

print("Generator:", GENERATOR_MODEL)

# ============================================================
# LOAD SQUAD
# ============================================================


def create_dataset(shuffle=True):
    print("\nLoading SQuAD...")
    dataset = load_dataset(CONFIG["dataset_name"], split="train")
    if shuffle:
        dataset = dataset.shuffle(seed=CONFIG["seed"])
    return dataset.select(range(CONFIG["n_documents"]))


dataset = create_dataset()
# print(dataset)


# ============================================================
# BUILD DOCUMENTS
# ============================================================
def create_database(dataset):
    def deduplicate(docs, chunks):
        unique_texts = set()
        unique_documents = []
        unique_chunks = []

        for doc, chunk in zip(docs, chunks):
            if doc.text in unique_texts:
                continue

            unique_texts.add(doc.text)
            unique_documents.append(doc)
            unique_chunks.append(chunk)

        return unique_documents, unique_chunks

    print("\nCreating documents...")

    splitter = SentenceSplitter(
        chunk_size=CONFIG["chunk_size"], chunk_overlap=CONFIG["chunk_overlap"]
    )

    documents = []

    chunks = []
    # print(f"\n\nPrzykładowy element z datasetu: {dataset[0]}\n\n")
    # print(f"\n\nKlucze: {list(dataset[0].keys())}\n\n")

    def clean_text(text: str):

        if not text:
            return ""

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    for row in tqdm(dataset):

        context = clean_text(row["context"])

        if len(context) < CONFIG["len_short_article_to_remove"]:
            continue

        source_doc = Document(
            text=f"Title: {row['title']}\n\n{context}",
            metadata={
                "question_id": row["id"],
                "title": row["title"],
                "question": row["question"],
                "gold_answer": row["answers"]["text"][0],
            },
        )

        nodes = splitter.get_nodes_from_documents([source_doc])

        for node in nodes:

            if len(node.text.split()) < CONFIG["len_short_chunk_to_remove"]:
                continue

            documents.append(node)

            chunks.append(
                {
                    "text": node.text,
                    "metadata": node.metadata,
                }
            )

    documents, chunks = deduplicate(documents, chunks)
    # print("Chunks:", len(chunks))
    # print("Documents:", len(documents))
    # print("\n przykładowy chunk:", chunks[-1])
    # print("\n przykładowy dokumnet:", documents[0])

    return documents, chunks


documents, chunks = create_database(dataset)

# ============================================================
# BM25
# ============================================================


def create_lexical_model(chunks):
    def bm25_tokenizer(text):
        doc = nlp(text)
        return [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and len(token.text) > 2
        ]

    print("\nBuilding BM25...")

    tokenized_chunks = [bm25_tokenizer(chunk["text"]) for chunk in tqdm(chunks)]

    bm25 = BM25Okapi(tokenized_chunks)

    print("BM25 ready")

    return bm25, bm25_tokenizer


sparse_retriever, sparse_tokenizer = create_lexical_model(chunks)


# ============================================================
# EMBEDDING MODEL
# ============================================================


def create_embedding_model():
    print("\nLoading embeddings...")

    embedding_model = HuggingFaceEmbedding(model_name=CONFIG["embedding_model"])

    Settings.embed_model = embedding_model
    return embedding_model


embedding_model = create_embedding_model()

# ============================================================
# CHROMA
# ============================================================


def build_chromadb(embedding_model):
    print("\nBuilding ChromaDB...")

    client = chromadb.PersistentClient(path="./chroma_db")

    # collection = client.get_or_create_collection("squad_rag")

    try:
        client.delete_collection("squad_rag")
    except Exception:
        pass

    collection = client.create_collection("squad_rag")

    vector_store = ChromaVectorStore(chroma_collection=collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embedding_model
    )

    dense_retriever = index.as_retriever(similarity_top_k=CONFIG["dense_top_k"])

    # print("DUPLICATE CHROMA DB CHECK")
    # print("Collection size:", collection.count())
    # print("Current documents:", len(documents))
    assert collection.count() == len(documents)
    print("Vector index ready")
    return dense_retriever


dense_retriever = build_chromadb(embedding_model)

# ============================================================
# RERANKER
# ============================================================


def create_reranker():
    print("\nLoading reranker...")
    reranker = CrossEncoder(CONFIG["reranker_model"])
    print("Reranker ready")
    return reranker


reranker = create_reranker()

# ============================================================
# GENERATOR
# ============================================================


def create_generator():
    print("\nLoading generator...")

    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)

    if not CONFIG["test_mode"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            GENERATOR_MODEL, quantization_config=bnb_config, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            GENERATOR_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    print("Generator ready")
    return generator


generator = create_generator()

# ============================================================
# PROMPT TEMPLATE
# ============================================================

PROMPT_TEMPLATE = """
You are a question answering assistant.

Answer ONLY using the provided context.

If the answer cannot be found in the context,
respond exactly:

I could not find the answer.

Context:
{context}

Question:
{question}

Answer:
"""

# ============================================================
# GENERATION
# ============================================================


def generate_answer(question, context):
    print("\n\nOTRZYMANY CONTEXT:", context)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    output = generator(prompt, max_new_tokens=128, do_sample=False)
    print("\n\nCO ZWRACA GENERATOR?", output)

    return output[0]["generated_text"]


# ============================================================
# VARIANT A
# NO RAG
# ============================================================


def answer_A(question):

    prompt = f"""
Question:
{question}

Answer:
"""

    output = generator(prompt, max_new_tokens=128, do_sample=False)

    return output[0]["generated_text"]


# ============================================================
# VARIANT B
# BM25
# ============================================================


def retrieve_sparse(question, chunks, sparse_retriever, sparse_tokenizer):
    query_tokens = sparse_tokenizer(question)
    bm25_scores = sparse_retriever.get_scores(query_tokens)
    bm25_top_idx = np.argsort(bm25_scores)[::-1][: CONFIG["bm25_top_k"]]

    docs = [chunks[i]["text"] for i in bm25_top_idx]

    return docs


def answer_B(question):

    docs = retrieve_sparse(question, chunks, sparse_retriever, sparse_tokenizer)

    context = "\n\n".join(docs)

    return generate_answer(question, context)


# ============================================================
# VARIANT C
# DENSE
# ============================================================


def retrieve_dense(question):

    nodes = dense_retriever.retrieve(question)

    docs = [node.text for node in nodes]

    return docs


def answer_C(question):

    docs = retrieve_dense(question)

    context = "\n\n".join(docs)

    return generate_answer(question, context)


# ============================================================
# VARIANT D
# DENSE + RERANK
# ============================================================


def retrieve_dense_rerank(question):

    dense_docs = retrieve_dense(question)

    pairs = [(question, doc) for doc in dense_docs]

    scores = reranker.predict(pairs)

    ranked = sorted(zip(dense_docs, scores), key=lambda x: x[1], reverse=True)

    docs = [x[0] for x in ranked[: CONFIG["rerank_top_k"]]]

    return docs


def answer_D(question):

    docs = retrieve_dense_rerank(question)

    context = "\n\n".join(docs)

    return generate_answer(question, context)


# ============================================================
# MAIN API
# ============================================================


def ask(question, variant="D"):

    variant = variant.upper()

    if variant == "A":
        return answer_A(question)

    elif variant == "B":
        return answer_B(question)

    elif variant == "C":
        return answer_C(question)

    elif variant == "D":
        return answer_D(question)

    else:
        raise ValueError(f"Unknown variant: {variant}")


# ============================================================
# DEMO
# ============================================================

question = "What is the capital of France?"

# print("\nQUESTION:")
# print(question)

print("Wariant A")
print(ask(question, variant="A"))
print("Wariant B")
print(ask(question, variant="B"))
print("Wariant C")
print(ask(question, variant="C"))
print("Wariant D")
print(ask(question, variant="D"))
