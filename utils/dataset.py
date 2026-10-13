from datasets import load_dataset


def create_dataset():
    text_corpus = []
    corpus_ids = []

    qa_pairs = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")[
        "test"
    ]

    corpus = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")["passages"]

    for item in corpus:
        corpus_ids.append(str(item["id"]))
        text_corpus.append(item["passage"])

    return qa_pairs, text_corpus, corpus_ids
