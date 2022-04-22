import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel 
from siumaai.losses.focal_loss import FocalLoss
from siumaai.losses.label_smoothing import LabelSmoothingCrossEntropy
from torch.nn.modules import BCEWithLogitsLoss

class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, num_label)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        # features_output1 = F.relu(features_output1)
        features_output1 = F.gelu(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2


class MRCForNer(nn.Module):
    def __init__(self, pretrain_model_path, dropout_rate, hidden_size, vocab_len=None):
        super(MRCForNer, self).__init__()

        self.model = AutoModel.from_pretrained(pretrain_model_path)

        self.start_outputs = nn.Linear(hidden_size, 1)
        self.end_outputs = nn.Linear(hidden_size, 1)
        self.span_embedding = MultiNonLinearClassifier(hidden_size * 2, 1, dropout_rate)
        self.criterion = BCEWithLogitsLoss()

        if vocab_len is not None:
            self.model.resize_token_embeddings(vocab_len)

    def forward(
            self, 
            input_ids, 
            token_type_ids, 
            attention_mask, 
            start_labels=None, 
            end_labels=None, 
            span_labels=None, 
            start_criterion_mask=None,
            end_criterion_mask=None,
            span_criterion_mask=None):

        # [batch_size, seq_len, hidden_size]
        sequence_output =self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]

        batch_size, seq_len, hid_size = sequence_output.size()

        # [batch_size, seq_len]
        start_logits = self.start_outputs(sequence_output).squeeze(-1)
        end_logits = self.end_outputs(sequence_output).squeeze(-1)

        # [batch_size, seq_len, seq_len, hidden_size]
        start_extend = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # [batch_size, seq_len, seq_len, hidden_size]
        end_extend = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # [batch, seq_len, seq_len, hidden*2]
        span_matrix = torch.cat([start_extend, end_extend], 3)
        # [batch, seq_len, seq_len]
        span_logits = self.span_embedding(span_matrix).squeeze(-1)
        
        outputs = (start_logits, end_logits, span_logits)

        # if start_labels is not None and end_labels is not None and span_labels is not None:
        if None not in [start_labels, end_labels, span_labels, start_criterion_mask, end_criterion_mask, span_criterion_mask]:

            # start_logits_mask = attention_mask.view(-1) == 1
            # end_logits_mask = attention_mask.view(-1) == 1

            # x_mask = attention_mask.unsqueeze(-2).expand(-1, seq_len, -1)
            # y_mask = attention_mask.unsqueeze(-1).expand(-1, -1, seq_len)
            # span_logits_mask = torch.triu(x_mask & y_mask, 0).view(-1) == 1

            start_criterion_mask = start_criterion_mask.view(-1).bool()
            end_criterion_mask = end_criterion_mask.view(-1).bool()
            span_criterion_mask = torch.triu(span_criterion_mask, 0).view(-1).bool()

            active_start_logits = start_logits.view(-1)[start_criterion_mask]
            active_end_logits = end_logits.view(-1)[end_criterion_mask]
            active_span_logits = span_logits.view(-1)[span_criterion_mask]

            active_start_labels = start_labels.view(-1)[start_criterion_mask]
            active_end_labels = end_labels.view(-1)[end_criterion_mask]
            active_span_labels = span_labels.view(-1)[span_criterion_mask]

            start_loss = self.criterion(active_start_logits, active_start_labels)
            end_loss = self.criterion(active_end_logits, active_end_labels)
            span_loss = self.criterion(active_span_logits, active_span_labels)
            outputs = (start_loss+end_loss+span_loss, ) + outputs

        return outputs

