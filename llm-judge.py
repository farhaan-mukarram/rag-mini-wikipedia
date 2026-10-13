import pandas as pd
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-E4B-it"

SYSTEM_PROMPT = "You are a helpful assistant. You need to act as a judge and compare two answers to questions, given the question and the two answers. Answer yes if they are similar and convey the same meaning, no otherwise. Respond only with `Yes` or `No`."


def format_prompt(ques, ans_1, ans_2):
    return f"Question: {ques}\nResponse 1: {ans_1}\nResponse 2:{ans_2}"


def main():

    # Load model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype="auto", device_map="auto"
    )

    for fileroot in ["res-base", "res-rag"]:
        df = pd.read_csv(f"{fileroot}.csv")
        res = []

        rows = list(df.iterrows())

        for row in tqdm(rows, desc=f"Judging results for {fileroot}..."):
            item = row[-1]
            question, answer, prediction = (
                item["question"],
                item["answer"],
                item["prediction"],
            )

            prompt = format_prompt(question, answer, prediction)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            # Process input
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            ).to(model.device)
            input_len = inputs["input_ids"].shape[-1]

            # Generate output
            outputs = model.generate(**inputs, max_new_tokens=16)
            response = processor.decode(
                outputs[0][input_len:], skip_special_tokens=False
            )

            # Parse output
            result = processor.parse_response(response, prefix=inputs["input_ids"])[
                "content"
            ]

            res.append(result)

        output_path = f"{fileroot}-judge.csv"
        df["judge_output"] = res
        df.to_csv(output_path)
        print(f"Judget results saved to {output_path}\n")


if __name__ == "__main__":
    main()
