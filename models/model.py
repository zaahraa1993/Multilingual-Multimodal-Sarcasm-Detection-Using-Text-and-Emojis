
import torch
import torch.nn as nn
from transformers import AutoModel

class MultimodalSarcasmModel(nn.Module):
    def __init__(self, text_model_name='xlm-roberta-base', emoji_vocab_size=256, emoji_emb_dim=16, hidden_size=128, num_classes=2):
        super().__init__()

        self.text_model = AutoModel.from_pretrained(text_model_name)

        # open layers:7-11
        for name, param in self.text_model.named_parameters():
            if any(name.startswith(f"encoder.layer.{i}") for i in [7,8,9,10,11]):
                param.requires_grad = True
            else:
                param.requires_grad = False

        text_hidden_size = self.text_model.config.hidden_size

        self.emoji_emb = nn.Embedding(emoji_vocab_size, emoji_emb_dim, padding_idx=0)

        
        self.norm = nn.LayerNorm(text_hidden_size + emoji_emb_dim)

        # 2 layers classifier 
        self.classifier = nn.Sequential(
            nn.Linear(text_hidden_size + emoji_emb_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, input_ids, attention_mask, emoji_ids):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = text_outputs.last_hidden_state[:, 0]

        emoji_embed = self.emoji_emb(emoji_ids)
        emoji_repr = emoji_embed.mean(dim=1)

        combined = torch.cat((cls_output, emoji_repr), dim=1)
        combined = self.norm(combined)

        outputs = self.classifier(combined)
        return outputs

