import re
import json
from pathlib import Path

def parse_whatsapp_chat(input_path, output_path, main_character, min_words=3):
    chat_lines = Path(input_path).read_text(encoding='utf-8').splitlines()
    
    # WhatsApp line pattern: "14/11/2021 11:56 - Name: Message"
    pattern = r'^(\d{1,2}/\d{1,2}/\d{2,4} \d{2}:\d{2}) - ([^:]+): (.+)$'
    
    messages = []
    for line in chat_lines:
        match = re.match(pattern, line)
        if match:
            _, speaker, text = match.groups()
            # Skip messages containing hidden media
            if "<Mídia oculta>" not in text:
                messages.append({"speaker": speaker.strip(), "text": text.strip()})
        elif messages and "<Mídia oculta>" not in line:
            messages[-1]["text"] += " " + line.strip()  # continuation line

    # Group messages by consecutive speaker
    grouped = []
    last_speaker = None
    buffer = []

    def flush():
        if last_speaker and buffer:
            grouped.append((last_speaker, buffer.copy()))
        buffer.clear()

    for msg in messages:
        if msg["speaker"] != last_speaker:
            flush()
            last_speaker = msg["speaker"]
        buffer.append(msg["text"])
    flush()

    # Instruction pairs: multiple lines of user -> multiple lines of Enzo
    samples = []
    i = 0
    while i < len(grouped) - 1:
        speaker_i, texts_i = grouped[i]
        speaker_j, texts_j = grouped[i + 1]

        if speaker_i != main_character and speaker_j == main_character:
            input_text = "\n".join(f"{speaker_i}: {t}" for t in texts_i)
            output_text = "\n".join(f"{speaker_j}: {t}" for t in texts_j)

            if len(input_text.split()) >= min_words and len(output_text.split()) >= min_words:
                samples.append({
                    "instruction": f"You are {main_character}. Respond as them.",
                    "input": input_text,
                    "output": output_text
                })
            i += 2  # skip both turns
        else:
            i += 1  # move to next

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(samples)} samples to {output_path}")

# Example usage
#parse_whatsapp_chat(
#    input_path="../data/whatsapp.txt",
#    output_path="../data/dataset.jsonl",
#    main_character="Enzo Bustamante"
#)