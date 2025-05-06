# 🎭 The Digital Twin Project:

Character based digital representation from pre-defined datasets.
Train a local Large Language Model (LLM) to speak like a specific character using a small, focused dataset. 


## 🛠️ Requirements

- Python 3.9+
- Unix-based system (Ubuntu recommended)
- One RTX 3090 GPUs (for 7B models)
- At least 24 GB RAM recommended
- [Hugging Face account](https://huggingface.co/join)

## 📦 Dependencies

```bash
# System packages
sudo apt update && sudo apt upgrade -y
sudo apt install git python3 python3-pip virtualenv -y

# Python virtual environment
python3 -m venv llm-env
source llm-env/bin/activate
pip install --upgrade pip
```

### 🔧 Python Packages

```bash
# Core ML libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# HuggingFace ecosystem
pip install transformers datasets accelerate peft bitsandbytes sentencepiece

# Training and optimization
pip install trl optimum
```

## 🔐 Hugging Face Authentication

```bash
huggingface-cli login
```

## 📁 Project Structure

```text
project-root/
├── data/
│   └── your_data.jsonl
├── scripts/
│   └── train.py
├── final_model/
├── README.md
└── requirements.txt
```

## 📚 Dataset Format

Each line in `your_data.jsonl` should look like this:

```json
{"text": "CHARACTER: 'To be, or not to be, that is the question.'"}
{"text": "CHARACTER: 'Thou art as wise as thou art beautiful.'"}
```

## 🚀 Running Training

```bash
python scripts/train.py
```

## 🧠 Inference (Example)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("final_model", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("final_model")

prompt = "CHARACTER: How do you feel about death?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## 📝 Notes

- You can train 7B models easily with 1× RTX 3090 using LoRA.
- Use `text-generation-webui` or `llama.cpp` for easier local chat interface after training.

## 📄 License

This project is licensed under the MIT License.
