from transformers import CLIPTokenizer

# Load the tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def preprocess_text(text: str):
    """
    Tokenizes and preprocesses the input text.
    """
    inputs = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=77)
    return inputs
