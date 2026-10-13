import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference import infer
from utils.dataset import create_dataset
from utils.chroma import init_chroma_client, init_collection, query_collection


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
    parser.add_argument("-m", "--mode", default="base")
    args = parser.parse_args()

    return args


def main():

    args = parse_args()
    mode = args.mode

    # args validation
    assert mode in ["rag", "base"]

    with_rag = mode == "rag"

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", device_map="auto")

    qa_pairs, corpus, doc_ids = create_dataset()

    rows = []

    for item in tqdm(qa_pairs, desc=f"Evaluating dataset in {mode} mode..."):
        question, answer, question_id = item["question"], item["answer"], item["id"]

        docs = None

        if with_rag:
            chroma_client = init_chroma_client()
            collection = init_collection(chroma_client, ids=doc_ids, documents=corpus)
            query_result = query_collection(collection, question)
            docs = query_result["documents"][0]

        prompt = format_prompt(question=question, context=docs)

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
