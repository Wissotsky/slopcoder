def main():
    from llama_cpp import Llama

    llm = Llama(
        model_path="./model_files/qwen2.5-coder-0.5b-instruct-q2_k.gguf",
        n_gpu_layers=-1, # Uncomment to use GPU acceleration
        # seed=1337, # Uncomment to set a specific seed
        n_ctx=2048 #n_ctx=32_768 # Uncomment to increase the context window
    )
    messages = []
    while True:
        user_input = input("You: ")
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        output = llm.create_chat_completion(
            messages=messages,
            repeat_penalty=1.2
        )

        print("Bot:", output)
        messages.append(output['choices'][0]['message'])
        



if __name__ == "__main__":
    main()
