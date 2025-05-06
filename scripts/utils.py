from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch

def load_model(model_path):
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print("Loading configuration...")
    config = PeftConfig.from_pretrained(model_path)

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto"
    )

    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    
    return model, tokenizer

def chat_loop(model, tokenizer, character_name, user_name, max_new_tokens=100):
    history = []
    while True:
        try:
            user_input = input(user_name + ": ")
            if not user_input.strip():  # Skip empty inputs
               continue
                    
            formatted_input = f"{user_name}: {user_input}"
            history.append(formatted_input)
                
            prompt = format_prompt(formatted_input, history, character_name)
            response = chat_with(model, tokenizer, prompt)
                
            if response:  # Only process if we got a response
                full_response = f"{character_name}: {response}"
                print(full_response)
                history.append(full_response)
            else:
                print("Não consegui gerar uma resposta. Tente novamente.")
                    
        except KeyboardInterrupt:
            print("\nConversa encerrada. Até mais!")
            break

def chat_with(model, tokenizer, prompt, max_new_tokens=100):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.5,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

def format_prompt(user_input, history, character_name, max_history=6):
    conversation = "\n".join(history[-max_history:])  # Keep last 3 exchanges (6 messages)
    return f"""### Instrução:
Você é {character_name}. Responda como ele. Mantenha suas respostas curtas e naturais.

### Entrada:
{conversation}
{user_input}

### Resposta:
{character_name}:"""