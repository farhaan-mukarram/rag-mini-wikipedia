SYSTEM_PROMPT = {
    "BASE": "You are a helpful assistant that helps the user answer questions. Keep your answers concise, use only short phrases and sentences.",
    "RAG": "You are a helpful assistant that helps the user answer questions, based on the provided evidence. Keep your answers concise, use only short phrases and sentences and answer ONLY based on the evidence/context provided. If the evidence/context is insufficient to answer the question, say so instead of guessing.",
}


def infer(model, tokenizer, prompt, with_rag=False):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT["RAG"] if with_rag else SYSTEM_PROMPT["BASE"],
        },
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )

    return answer
