import torch
from torch import nn
from transformers import AutoModel
from siumaai.losses.multilabel_categorical_crossentropy import multilabel_categorical_crossentropy


class GlobalPointerForNer(nn.Module):
    def __init__(self, pretrain_model_path, hidden_size, num_labels, inner_dim, RoPE=True, vocab_len=None):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrain_model_path)
        self.num_labels = num_labels
        self.inner_dim = inner_dim
        self.dense = nn.Linear(hidden_size, self.num_labels * self.inner_dim * 2)

        self.RoPE = RoPE

        if vocab_len is not None:
            self.model.resize_token_embeddings(vocab_len)
    
    
    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings
        
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, criterion_mask=None):
        self.device = input_ids.device
        
        context_outputs = self.model(input_ids, attention_mask, token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, num_labels*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, num_labels, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, num_labels, inner_dim)
        qw, kw = outputs[...,:self.inner_dim], outputs[...,self.inner_dim:] # TODO:修改为Linear获取？

        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None,::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[...,::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[...,::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
            
        # logits:(batch_size, num_labels, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.num_labels, seq_len, seq_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.num_labels, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        logits = logits*pad_mask - (1-pad_mask)*1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1) 
        logits = logits - mask * 1e12
        
        logits = logits/self.num_labels**0.5

        if labels is not None and criterion_mask is not None:

            criterion_mask = torch.triu(criterion_mask, 0).view(-1).bool()
            active_logits = logits.view(-1)[criterion_mask]
            active_labels = labels.view(-1)[criterion_mask]
            loss = multilabel_categorical_crossentropy(active_logits, active_labels)

            # loss = multilabel_categorical_crossentropy(
            #         logits.view(batch_size*self.num_labels, -1), 
            #         labels.view(batch_size*self.num_labels, -1))
            return loss, logits
        else:
            return (logits,)



        
        # return logits/self.num_labels**0.5
