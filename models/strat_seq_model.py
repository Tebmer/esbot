# coding=utf-8
# copied from bart

from turtle import forward
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F


class StratSeqModel(nn.Module):
    def __init__(self, config):
        super(StratSeqModel, self).__init__()
        
        # hyper parameters and others
        self.num_hiddens = config['num_hiddens']
        self.input_size = config['input_size']
        self.num_strat = config['num_strat']
        self.model_type = config['model_type']  
        self.num_layers = config['num_layers']      
        self.bidirectional = config['bidirectional']
        self.add_context_method = config['add_context_method']
        
        # model part
        # self.num_strat + 1: the id of additional `begin_strat_id`,
        # but when predicting, we do not consider it.
        self.strat_embedding = nn.Embedding(self.num_strat + 1, self.input_size)
        if self.model_type == 'rnn':
            self.model_class = nn.RNN
        elif self.model_type == 'lstm':
            self.model_class = nn.LSTM
        elif self.model_type == 'gru':
            self.model_class = nn.GRU
        
        self.rnn = self.model_class(
            self.input_size, 
            self.num_hiddens, 
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True
        )
            
        self.dense_seq = nn.Linear(self.num_hiddens, self.num_strat)
        self.dense_cat = nn.Linear(2 * self.num_hiddens, self.num_strat)
        self.dense_context = nn.Linear(self.num_hiddens, self.num_strat)

        # loss function
        self.loss_func = nn.CrossEntropyLoss()
        
        self._init_params()
       
    # initialize weight 
    def _init_params(self):
        nn.init.xavier_normal_(self.strat_embedding.weight)
        nn.init.xavier_normal_(self.dense_seq.weight)
        nn.init.constant_(self.dense_seq.bias, 0)
        nn.init.xavier_normal_(self.dense_cat.weight)
        nn.init.constant_(self.dense_cat.bias, 0)
        nn.init.xavier_normal_(self.dense_context.weight)
        nn.init.constant_(self.dense_context.bias, 0)
        
    def forward(self, strat_seq, strat_seq_len, context_hidden_state=None, strat_id=None):
        strat_embs = self.strat_embedding(strat_seq)
        output, _ = self.rnn(strat_embs)    #h: [B, L, 2*H] if bidirecitonal, else [B, L, H]
        
        if self.bidirectional:
            output = output.view(strat_seq.size(0), strat_seq.size(1), 2, self.num_hiddens) #[B, L, 2, H]
            output = output.sum(dim=2)  # [B, L, H]
            
        # Get the hiddens of target postion
        # Here, we want to get the target postional hiddens according to the `seq_len`.
        # Thus, we first enable them the same dimensions and then use `gather` function.
        h = output.permute(0, 2, 1) # [B, H, L]
        target_pos = (strat_seq_len - 1).unsqueeze(-1).unsqueeze(-1)     # [B, 1, 1]
        target_pos = target_pos.repeat(1, self.num_hiddens, 1)  # [B, H, 1]
        h = torch.gather(h, dim=-1, index=target_pos)   # [B, H, 1]
        h = h.permute(0, 1, 2).squeeze(-1)  # [B, H]
        
        if context_hidden_state is not None:
            # TODO: add a special [CLS] token for classify strategy token.
            context_hidden_state = torch.mean(context_hidden_state, dim=1)
            
            if self.add_context_method == 1:    # Add hidden state 
                h = (context_hidden_state + h) / 2
                logits = self.dense_seq(h) 
            elif self.add_context_method == 2:  # Concatenate hidden state
                h = torch.cat([context_hidden_state, h], dim=-1)
                logits = self.dense_cat(h)  
            elif self.add_context_method == 3:  # Add logits 
                logits_seq = self.dense_seq(h)
                logits_context = self.dense_context(context_hidden_state)
                logits = (logits_seq + logits_context) / 2
            else:       # Do not use context information.
                logits = self.dense_seq(h)
                        
        loss = 0
        if strat_id is not None:
            loss = self.loss_func(logits, strat_id)
        
        pred = self.predict_strategy(logits)
        
        return logits, pred, loss
    
    def predict_strategy(self, logits):
        pred_top1 = torch.argmax(logits, dim=1)
        pred_top3 = torch.topk(logits, k=3, dim=-1)[1]
        return {"pred_top1": pred_top1, "pred_top3": pred_top1}
        
        
        