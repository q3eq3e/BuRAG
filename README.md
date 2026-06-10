# BuRAG - Broad Understanding Retrieval-Augmented Generation

# Temat: RAG - pytania i odpowiedzi w oparciu o bazę dokumentów

## Autorzy: Jakub Bagiński i Maciej Borkowski

## Warianty:
- A – bez RAG
- B – RAG + sparse retrieval
- C – RAG + dense retrieval
- D – RAG + dense retrieval + re-ranking

## Wykorzystywane technologie:
- model: 		Llama 3 (8B Instruct)
- Embeddings:	bge-small-en (HuggingFaceEmbeddings)
- Reranker: 	Cross-encoder MS MARCO MiniLM L-6 v2
- Sparse retrieval: BM25
- Baza wektorowa: ChromaDB
- Orkiestrator:	LlamaIndex

## Dataset
rajpurkar/squad (Hugging Face) – 100k par pytanie/odpowiedź na bazie artykułów Wikipedii


## Planowane eksperymenty:

### Eksperyment 1 – Wpływ RAG na jakość odpowiedzi
Pytanie badawcze

Czy dodanie retrieval i re-ranking poprawia jakość generowanych odpowiedzi?

Porównywane warianty

A vs B vs C vs D

Procedura

Dla każdego pytania:

- wygeneruj odpowiedź
- porównaj z ground truth

Metryki:
- ROUGE-L
- chrF

Oczekiwana obserwacja

A << B < C < D



### Eksperyment 2 – Jakość wyszukiwania dokumentów
Pytanie badawcze

Która metoda wyszukiwania najlepiej odnajduje właściwy fragment?

Porównywane warianty

B vs C vs D

Metryka:
- Recall@k

Procedura

Sprawdzamy czy poprawny chunk znajduje się w top-k, dla różnych k.

Oczekiwana obserwacja

BM25 < Dense < Dense + reranker

### Eksperyment 3 – Wpływ liczby pobieranych fragmentów (top-k)

Pytanie badawcze

Ile fragmentów kontekstu powinien dostać LLM?

Testowane wartości

k = 3, 5, 10

Metryki
- Recall@k (retrieval)
- ROUGE / chrF (generacja)
