import re
import emoji
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoTokenizer



# 1. cleaning text 
def clean_text(text):
    text = re.sub(r"http\S+", "", text)         
    text = re.sub(r"@\w+", "", text)           
    text = re.sub(r"#", "", text)
    text = re.sub(r"@user", "", text)
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()    
    return text

# 2.extract emoji from text 
def extract_emojis(text):
    return " ".join([c for c in text if c in emoji.EMOJI_DATA])

# 3. change emojies to indices
def build_emoji_vocab(emojis_list):
    all_emojis = set(" ".join(emojis_list).split())
    emoji2id = {e: i+1 for i, e in enumerate(sorted(all_emojis))}  
    emoji2id['<PAD>'] = 0
    return emoji2id

def encode_emojis(emoji_str, emoji2id, max_len=5):
    ids = [emoji2id.get(e, 0) for e in emoji_str.split()]
    return ids[:max_len] + [0] * (max_len - len(ids))  

#  4.dataset
class SarcasmDataset(Dataset):
    def __init__(self, df, tokenizer, emoji2id, max_text_len=64, max_emoji_len=5):
        self.texts = df['clean_text'].tolist()
        self.emoji_ids = df['emoji_ids'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_emoji_len = max_emoji_len
        self.emoji2id = emoji2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        emoji_ids = self.emoji_ids[idx]
        label = self.labels[idx]

        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'emoji_ids': torch.tensor(emoji_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
train_df = pd.read_parquet("/content/drive/MyDrive/train-00000-of-00001.parquet")
val_df   = pd.read_parquet("/content/drive/MyDrive/validation-00000-of-00001.parquet")
test_df  = pd.read_parquet("/content/drive/MyDrive/test-00000-of-00001.parquet")
train_df['clean_text'] = train_df['text'].apply(clean_text)
train_df['emojis'] = train_df['text'].apply(extract_emojis)

val_df['clean_text'] = val_df['text'].apply(clean_text)
val_df['emojis'] = val_df['text'].apply(extract_emojis)


emoji2id = build_emoji_vocab(train_df['emojis'])


train_df['emoji_ids'] = train_df['emojis'].apply(lambda e: encode_emojis(e, emoji2id))
val_df['emoji_ids'] = val_df['emojis'].apply(lambda e: encode_emojis(e, emoji2id) )

train_df[['clean_text', 'emojis', 'emoji_ids', 'label']].head()



test_df['clean_text'] = test_df['text'].apply(clean_text)
test_df['emojis'] = test_df['text'].apply(extract_emojis)
test_df['emoji_ids'] = test_df['emojis'].apply(lambda e: encode_emojis(e, emoji2id))
