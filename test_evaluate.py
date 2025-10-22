from preprocessing import test_df,SarcasmDataset,emoji2id,tokenizer
from torch.utils.data import DataLoader
from lit_model import LitSarcasmModel
from evaluate import evaluate_model 


test_data = SarcasmDataset(test_df, tokenizer, emoji2id)
test_loader = DataLoader(test_data, batch_size=32)


model = LitSarcasmModel.load_from_checkpoint("/content/logs/sarcasm/version_7/checkpoints/epoch=8-step=810.ckpt", emoji_vocab_size=len(emoji2id), class_weights=class_weights)
model.eval()


evaluate_model(model, test_loader, class_names=["non-sarcastic", "sarcastic"])

