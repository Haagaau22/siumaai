from torch import nn
from transformers import AutoModel 
from siumaai.layers.crf import CRF




class CrfForNer(nn.Module):
    def __init__(self, pretrain_model_path, num_labels, dropout_rate, hidden_size, vocab_len=None, pad_id=-100):
        super(CrfForNer, self).__init__()

        self.model = AutoModel.from_pretrained(pretrain_model_path)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)
        self.pad_id = pad_id
        if vocab_len is not None:
            self.model.resize_token_embeddings(vocab_len)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs =self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            # mask = attention_mask & (labels != self.pad_id).long()
            # loss = self.crf(emissions=logits, tags=labels, mask=mask)
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,) + outputs
        else:
            crf_outputs = self.crf.decode(emissions=logits, mask=attention_mask)
            outputs = (crf_outputs, ) + outputs
        return outputs
