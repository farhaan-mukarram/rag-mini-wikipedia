import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def init_chroma_client():
    client = chromadb.PersistentClient()

    return client


def init_collection(client, ids=None, documents=None, embedding_fn=None):
    if embedding_fn is None:
        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2", device="cuda", normalize_embeddings=False
        )

    collection = client.get_or_create_collection(
        name="mini-wiki", embedding_function=embedding_fn
    )

    if ids is not None and documents is not None:
        collection.add(ids=ids, documents=documents)

    return collection


def query_collection(collection, query):
    results = collection.query(query_texts=[query], n_results=5)

    return results
