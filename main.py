import argparse

import pandas as pd
from sentence_transformers import CrossEncoder
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference import infer
from utils.chroma import init_chroma_client, init_collection, query_collection
from utils.dataset import create_dataset


def format_prompt(question: str, context: list[str] | None = None):
    prompt = None
    if context is None:
        prompt = f"Answer the following question: {question}"

    else:
        context_str = "\n".join(context)

        prompt = f"Answer the following question: {question} based on the context: {context_str}"

    return prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mode", default="base", choices=["base", "rag", "rag-reranked"]
    )
    args = parser.parse_args()

    return args


def main():

    args = parse_args()
    mode = args.mode

    with_rag = mode != "base"
    with_reranker = mode == "rag-reranked"

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", device_map="auto")

    reranker = None

    qa_pairs, corpus, doc_ids = create_dataset()

    rows = []

    chroma_client = None
    collection = None

    if with_rag:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        chroma_client = init_chroma_client()
        collection = init_collection(chroma_client, ids=doc_ids, documents=corpus)

    for item in tqdm(qa_pairs, desc=f"Evaluating dataset in {mode} mode..."):
        question, answer, question_id = item["question"], item["answer"], item["id"]

        docs = None
        reranked_docs = None

        if with_rag:
            query_result = query_collection(collection, question)
            docs = query_result["documents"][0]
            reranked_docs = docs

            if with_reranker:
                ranks = reranker.rank(question, docs)

                # pick top 5 ranked docs
                top_5_ranks = ranks[:5]
                reranked_docs = [docs[rank["corpus_id"]] for rank in top_5_ranks]

        prompt = format_prompt(question=question, context=reranked_docs)

        prediction = infer(
            model=model, tokenizer=tokenizer, prompt=prompt, with_rag=with_rag
        )

        rows.append([question_id, question, answer, prediction])

    df = pd.DataFrame(rows, columns=["question_id", "question", "answer", "prediction"])
    output_path = f"res-{mode}.csv"
    df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
