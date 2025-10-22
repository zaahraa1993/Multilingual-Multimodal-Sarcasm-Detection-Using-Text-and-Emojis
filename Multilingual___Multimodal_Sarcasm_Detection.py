import json
from transformers import AutoTokenizer
from preprocessing import emoji_encode,clean_txt,get_emojis,emoji2id
from lit_model import LitSarcasmModel
import torch




tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
with open("emoji2id.json", "r", encoding="utf-8") as f:
    emoji2id = json.load(f)

MAX_EMOJI_LEN=5
MODEL_PATH = "/content/drive/MyDrive/epoch=8-step=810.ckpt"

# Prediction Function
def predict_tweet(tweet):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Clean and preprocess
    clean_txt = clean_txt(tweet)
    emoji_str = get_emojis(tweet)
    emoji_ids = emoji_encode(emoji_str, emoji2id, max_len=MAX_EMOJI_LEN)

    # Tokenize
    encoded = tokenizer(clean_txt, return_tensors="pt", padding="max_length", truncation=True, max_length=64)

    ids = encoded["ids"].to(device)
    mask = encoded["mask"].to(device)
    emoji_tensor = torch.tensor([emoji_ids], dtype=torch.long).to(device)

    # Load model
    model = LitSarcasmModel.load_from_checkpoint(MODEL_PATH, emoji_vocab_size=len(emoji2id))
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(ids, mask, emoji_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return pred, probs.squeeze().tolist()

if __name__ == "__main__":
    tweet = input("Enter a tweet: ")

    label, probs = predict_tweet(tweet)
    print("\n Prediction:", "Sarcastic " if label == 1 else "Not Sarcastic ")
    print(f" Probabilities → Non-sarcastic: {probs[0]:.2f}, Sarcastic: {probs[1]:.2f}")