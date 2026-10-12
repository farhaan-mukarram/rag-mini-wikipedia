import pandas as pd
from datasets import load_dataset
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference import infer
from tqdm import tqdm


def main():

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", device_map="auto")

    qa_pairs = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")[
        "test"
    ]

    rows = []

    for item in tqdm(qa_pairs, desc="Evaluating..."):
        question, answer, question_id = item["question"], item["answer"], item["id"]
        prompt = f"Answer the following question: {question}"
        prediction = infer(model, tokenizer, prompt)
        rows.append([question_id, question, answer, prediction])

    df = pd.DataFrame(rows, columns=["question_id", "question", "answer", "prediction"])
    df.to_csv("res-base.csv", index=False)


if __name__ == "__main__":
    main()
