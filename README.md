# 🎭 Digital Twin Project

A character-based digital representation system that uses Large Language Models (LLMs) to create AI personas based on chat data. This project allows you to train a local LLM to mimic a specific person's communication style using their chat history.

## 🎯 Features

- Train a Mistral-7B model to mimic a specific person's communication style
- Process WhatsApp chat exports to create training data
- Interactive chat interface to interact with the trained model
- LoRA fine-tuning for efficient training on consumer hardware
- 4-bit quantization for reduced memory requirements

## 🛠️ Requirements

- Python 3.9+
- CUDA-capable GPU (RTX 3090 or better recommended)
- At least 24 GB RAM
- [Hugging Face account](https://huggingface.co/join)

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/digitaltwin.git
cd digitaltwin
```

2. Create and activate virtual environment:
```bash
python -m venv llm-env
source llm-env/bin/activate  # On Windows: .\llm-env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
digitaltwin/
├── data/
│   ├── whatsapp.txt        # Raw WhatsApp chat export
│   └── dataset.jsonl       # Processed training data
├── scripts/
│   ├── data_builder.py     # Data processing utilities
│   └── utils.py           # Model loading and chat functions
├── notebooks/
│   └── digitaltwin.ipynb   # Training and chat notebook
├── models/                 # Saved model checkpoints
├── logs/                   # Training logs
└── requirements.txt        # Project dependencies
```

## 🚀 Usage

### 1. Prepare Your Data

Export your WhatsApp chat and place it in `data/whatsapp.txt`. The chat should be in the standard WhatsApp export format.

### 2. Process the Data

Run the data processing script to convert the WhatsApp chat into training data:

```python
from scripts.data_builder import parse_whatsapp_chat

parse_whatsapp_chat(
    input_path="data/whatsapp.txt",
    output_path="data/dataset.jsonl",
    main_character="Your Name"
)
```

### 3. Train the Model

Open and run the `notebooks/digitaltwin.ipynb` notebook. The notebook will:
- Load the Mistral-7B model
- Process your training data
- Fine-tune the model using LoRA
- Save the trained model

### 4. Chat with Your Digital Twin

After training, you can interact with your digital twin using the chat interface:

```python
from scripts.utils import load_model, chat_loop

model, tokenizer = load_model("./models/your_model")
chat_loop(model, tokenizer, "Your Name", "User")
```

## 🔧 Technical Details

- Uses Mistral-7B as the base model
- Implements LoRA for efficient fine-tuning
- 4-bit quantization for reduced memory usage
- Custom tokenizer for optimal training
- Interactive chat loop with conversation history

## 📝 Notes

- Training requires approximately 24GB of VRAM
- The model is fine-tuned using LoRA to reduce memory requirements
- Chat history is maintained during conversations for context
- The model is saved in a format compatible with Hugging Face's transformers library

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
