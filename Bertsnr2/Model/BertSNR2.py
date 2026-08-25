# BERTSNR with DNABERT2
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch

class BERTSNR(nn.Module):
    def __init__(self):
        super(BERTSNR, self).__init__()
        bert_path = "/home/ubuntu/.cache/huggingface/hub/models--zhihan1996--DNABERT-2-117M/snapshots/7bce263b15377fc15361f52cfab88f8b586abda0"
        self.bert = AutoModel.from_pretrained(bert_path, trust_remote_code=True)
        self.dropout = nn.Dropout(0.2)
        self.classifier_token = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )
        self.classifier_sequence = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, offset_mapping):
        # offset_mapping is passed in but NOT sent to BERT
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = output[0]  # (batch, seq_len, 768)

        # --- Sequence classification: CLS token ---
        cls = hidden_states[:, 0, :]
        cls = self.dropout(cls)
        logits_sequence = self.classifier_sequence(cls)

        # --- Token classification: nucleotide-level via offset mapping ---
        batch_size = hidden_states.shape[0]
        all_nucleotide_logits = []

        for b in range(batch_size):
            offsets = offset_mapping[b]
            embeddings = hidden_states[b]
            nucleotide_embeddings = []

            for i in range(1, len(offsets)):
                start, end = offsets[i][0].item(), offsets[i][1].item()
                if start == 0 and end == 0:
                    break  # padding token
                token_emb = embeddings[i]
                for _ in range(end - start):
                    nucleotide_embeddings.append(token_emb)

            if nucleotide_embeddings:
                nuc_tensor = torch.stack(nucleotide_embeddings)
            else:
                nuc_tensor = embeddings.new_zeros((0, 768))

            nuc_tensor = self.dropout(nuc_tensor)
            logits = self.classifier_token(nuc_tensor)
            all_nucleotide_logits.append(logits)

        return logits_sequence, all_nucleotide_logits