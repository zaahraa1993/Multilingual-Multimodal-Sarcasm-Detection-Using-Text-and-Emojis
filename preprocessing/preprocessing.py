import re
import emoji
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer




# 1. clean tweet
def clean_txt(t):
    t = re.sub(r"http\S+", "", t)         
    t = re.sub(r"@\w+", "", t)           
    t = t.replace("#", "")            
    t = t.replace("\n", " ")
    t = re.sub(r"@user", "", t)         
    return t.strip()

# 2.extract emoji
def get_emojis(x):
    out = []
    for ch in x:
        if ch in emoji.EMOJI_DATA:
            out.append(ch)
    return " ".join(out)

# 3. change to indices
def emoji_dict(emojis_list):
    all_emojis = set()
    emoji2id = {}
    for seq in emojis_list:
        for e in seq.split():
            all_emojis.add(e)
    sorted_emojis = sorted(all_emojis)
    for index, emoji_char in enumerate(sorted_emojis):
    # start index from 1 (0 used for padding)
        emoji2id[emoji_char] = index + 1
    emoji2id["<PAD>"] = 0

    return emoji2id

def emoji_encode(emoji_str, emoji2id, max_len=5):

    # split the emoji string into individual emojis
    emoji_chars = emoji_str.split()
    
    # map each emoji to its ID, use 0 if not found
    ids = []
    for e in emoji_chars:
        ids.append(emoji2id.get(e, 0))

    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    
    return ids
 

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
