from torch import nn
from transformers import AutoModel 
from siumaai.losses.focal_loss import FocalLoss
from siumaai.losses.label_smoothing import LabelSmoothingCrossEntropy




class SpanForNer(nn.Module):
    def __init__(self, pretrain_model_path, num_labels, dropout_rate, hidden_size, vocab_len=None, loss_type=None):
        super(SpanForNer, self).__init__()

        self.model = AutoModel.from_pretrained(pretrain_model_path)
        self.dropout = nn.Dropout(dropout_rate)
        self.start_fc = nn.Linear(hidden_size, num_labels)
        self.end_fc = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        
        if loss_type == 'focal':
            self.criterion = FocalLoss()
        elif loss_type == 'ls_ce':
            self.criterion = LabelSmoothingCrossEntropy()
        else:
            self.criterion = nn.CrossEntropyLoss()

        if vocab_len is not None:
            self.model.resize_token_embeddings(vocab_len)

    def forward(self, input_ids, token_type_ids, attention_mask, start_labels=None, end_labels=None):
        outputs =self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        start_logits = self.start_fc(sequence_output)
        end_logits = self.end_fc(sequence_output)
        outputs = (start_logits, end_logits)

        if start_labels is not None and end_labels is not None:
            active_loss = attention_mask.view(-1) == 1

            active_start_logits = start_logits.view(-1, self.num_labels)[active_loss]
            active_end_logits = end_logits.view(-1, self.num_labels)[active_loss]

            active_start_labels = start_labels.view(-1)[active_loss]
            active_end_labels = end_labels.view(-1)[active_loss]

            start_loss = self.criterion(active_start_logits, active_start_labels)
            end_loss = self.criterion(active_end_logits, active_end_labels)
            outputs = (start_loss+end_loss, ) + outputs


        return outputs
