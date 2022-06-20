# coding=utf-8
# copied from bart


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import BaseModel
from models.strat_seq_model import StratSeqModel
from src.transformers.generation_utils import top_k_top_p_filtering
from src.transformers.models.blenderbot_small import (BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration,)
# from src.transformers import (BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration,)
from src.transformers.modeling_outputs import (BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput,)
from .PARAMS import SAMPLE, TEMPERATURE


class Model(BaseModel, BlenderbotSmallForConditionalGeneration):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)
        self.strat_seq_model = None # Refer to the class `StratSeqModel`

    def forward(
         self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        validation=False,
        **kwargs
    ):
        
        
        encoded_info = kwargs
    
        assert self.toker is not None
        assert (self.training or validation) == (labels is not None)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and not validation: # inference
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

                
        if encoder_outputs is None: # Avoid repeatly computing strategy or encoding when generating.
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        # Strategy Prediction and Embeddings
        strat_seq_input = encoded_info.get('strat_seq_input', None)
        # When calling generate(), forward() also be called by generate(),
        # and in this case, some useless parameters in `kwargs` is removed.
        strat_loss = None
        if strat_seq_input is not None and self.strat_seq_model is not None:
            _, pred_strategy, strat_loss = self.strat_seq_model(context_hidden_state=encoder_outputs.last_hidden_state, 
                                                    **strat_seq_input)
            encoder_strategy_embeds = self.strat_seq_model.strat_embedding(strat_seq_input['strat_id'].unsqueeze(-1))
            # encoder_strategy_embeds = self.strat_seq_model.strat_embedding(pred_strategy['pred_top1'].unsqueeze(-1))
            # add `encoder_strategy_embeds` to `encoder_outputs`
            encoder_outputs.encoder_strategy_embeds = encoder_strategy_embeds
            
        
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_strategy_embeds=encoder_outputs.encoder_strategy_embeds,
        )
            
        lm_logits = self.lm_head(decoder_outputs[0]) + self.final_logits_bias
        
        masked_lm_loss = None
        if labels is not None:
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')   # [B * L]
            loss = loss.view(labels.size(0), labels.size(1))    # [B, L]
            label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)    # [B, 1], do not consider loss of padding token.
            masked_lm_loss = torch.sum(loss) / torch.sum(label_size)    # masked_lm_loss: average lm loss.
            ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))

        
            
        if not self.training and not validation: # inference
            if not return_dict:
                output = (lm_logits,) + decoder_outputs[1:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )

        elif self.training: # training
            assert not validation
            res = {'all': masked_lm_loss, 'ppl': ppl_value, 'strat_loss': strat_loss}
            # TODO: modify this loss
            # res = {'all': strat_loss, 'ppl': ppl_value, 'strat_loss': strat_loss}
            return res

        else: # validation
            assert not self.training
            return loss, label_size, strat_loss
    
    # TODO: merge this function into the class `StratSeqModel`.
    def predict_strategy(self, logits, encoded_info):
        assert not self.training
        strat_id = encoded_info.get('strat_id', None)
    
        if strat_id is not None:
            pred = strat_id
        else:
            if SAMPLE:
                filtered_logits = top_k_top_p_filtering(logits / TEMPERATURE, top_p=0.9)
                pred = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(-1)
            else:
                pred = torch.argmax(logits, dim=-1)

            
        pred_top1 = torch.topk(logits, k=1, dim=-1)[1]
        pred_top3 = torch.topk(logits, k=3, dim=-1)[1]

        # ORACLE STRATEGY: use frequency of occurrence of strategies as logits 
        # strategey_freqs = torch.tensor([20.9, 5.9, 7.8, 9.4, 16.1, 15.6, 6.1, 19.2], dtype=torch.float)
        # oracle_logits = strategey_freqs.expand_as(logits) 
        # pred_top1 = torch.multinomial(oracle_logits, num_samples=1)
        # pred_top3 = torch.multinomial(oracle_logits, num_samples=3)
        # pred = pred_top1
        
        encoded_info.update({
            'pred_strat_id': pred,
            'pred_strat_id_top1': pred_top1,
            'pred_strat_id_top3': pred_top3,
            'pred_strat_id_dist': F.softmax(logits, dim=-1)
        })
    
    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        return_dict=None,
        **kwargs
    ):
        
        assert not self.training
        assert self.toker is not None
        
        encoded_info = kwargs
        assert decoder_input_ids.size(1) == 1
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict



        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        
        # predict strategy 
        strat_seq_input = encoded_info.get('strat_seq_input')
        
        if self.strat_seq_model is not None:
            logits, pred_strategy, _ = self.strat_seq_model(context_hidden_state=encoder_outputs.last_hidden_state, **strat_seq_input)
            self.predict_strategy(logits, encoded_info=encoded_info)
        
        # encoder_strategy_embeds = self.strat_seq_model.strat_embedding(strat_seq_input['strat_id'].unsqueeze(-1))
        encoder_strategy_embeds = self.strat_seq_model.strat_embedding(pred_strategy['pred_top1'].unsqueeze(-1))
        encoder_outputs.encoder_strategy_embeds = encoder_strategy_embeds 
        
        assert 'max_length' in kwargs
        kwargs['max_length'] = kwargs['max_length'] + decoder_input_ids.size(1)
        kwargs['use_cache'] = True
        
        if len(self.toker) > self.toker.vocab_size:
            bad_words_ids = [[i] for i in range(self.toker.vocab_size, len(self.toker))]
            kwargs['bad_words_ids'] = bad_words_ids
        
        generations = super().generate(
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            **kwargs
        )
        return encoded_info, generations[:, decoder_input_ids.size(1):]
