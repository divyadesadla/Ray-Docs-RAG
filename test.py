from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Guess model class based on model name (or you could define it manually per model)
    if "t5" in model_name.lower() or "mbart" in model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model_type = "seq2seq"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model_type = "causal"
    
    return tokenizer, model, model_type

def ask_question(query, tokenizer, model, model_type):
    prompt = f"Answer the following question in detail:\n\n{query}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    if model_type == "seq2seq":
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
        )
    else:  # causal model
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,    # causal models like OpenChat work better with sampling
            top_p=0.95,
            temperature=0.7
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Models to compare
    models_to_test = [
        "google/flan-t5-large",
        "openchat/openchat-3.5-0106"
        # Don't use mistral here unless you have access
    ]

    print("\n🚀 Multi-model QA: Ask a question and compare answers across models.\n")

    while True:
        user_input = input("❓ Your question (or 'exit'): ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        for model_name in models_to_test:
            print(f"\n🧠 {model_name}")
            try:
                tokenizer, model, model_type = load_model(model_name)
                answer = ask_question(user_input, tokenizer, model, model_type)
                print(f"🤖 Answer: {answer}")
            except Exception as e:
                print(f"⚠️ Error loading/generating with {model_name}: {e}")
