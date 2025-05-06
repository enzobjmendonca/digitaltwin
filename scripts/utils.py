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

class Chat:
    def __init__(self, model, tokenizer, character_name, user_name, max_new_tokens=100, top_k=50, top_p=0.95, temperature=0.5, repetition_penalty=1.2):
        self.model = model
        self.tokenizer = tokenizer
        self.character_name = character_name
        self.user_name = user_name
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty


    def chat_loop(self):
        history = []
        while True:
            try:
                user_input = input(self.user_name + ": ")
                if not user_input.strip():  # Skip empty inputs
                    continue
                        
                formatted_input = f"{self.user_name}: {user_input}"
                history.append(formatted_input)
                    
                prompt = self.format_prompt(formatted_input, history, self.character_name)
                response = self.chat_with(prompt)
                    
                if response:  # Only process if we got a response
                    full_response = f"{self.character_name}: {response}"
                    print(full_response)
                    history.append(full_response)
                else:
                    print("Could not generate a response. Please try again.")
                        
            except KeyboardInterrupt:
                print("\nConversation ended. Goodbye!")
                break

    def chat_with(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response[len(prompt):].strip()

    def format_prompt(self, user_input, history, character_name, max_history=6):
        conversation = "\n".join(history[-max_history:])  # Keep last 3 exchanges (6 messages)
        return f"""### Instruction:
    You are {character_name}. Respond as them. Keep your responses short and natural.

    ### Input:
    {conversation}
    {user_input}

    ### Response:
    {character_name}:"""