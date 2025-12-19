# modeling_pref_qwen3.py
import torch
from torch import nn
from typing import Optional, Union, List
from transformers import Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

class PrefQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # 从 config 中读取偏好维度，默认为 2
        self.pref_dim = getattr(config, "pref_dim", 2)
        # 扩展 LM Head: hidden_size + pref_dim
        self.lm_head = nn.Linear(config.hidden_size + self.pref_dim, config.vocab_size, bias=False)
        self.config.tie_word_embeddings = False

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, 
                inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, 
                output_hidden_states=None, return_dict=None, pref_vec=None, **kwargs):
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, 
                             past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, 
                             output_attentions=output_attentions, output_hidden_states=output_hidden_states, 
                             return_dict=return_dict, **kwargs)

        hidden_states = outputs[0]
        if pref_vec is not None:
            if pref_vec.dim() == 2:
                pref_vec = pref_vec.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
            combined_states = torch.cat([hidden_states, pref_vec], dim=-1)
        else:
            zeros = torch.zeros((*hidden_states.shape[:-1], self.pref_dim), device=hidden_states.device, dtype=hidden_states.dtype)
            combined_states = torch.cat([hidden_states, zeros], dim=-1)

        logits = self.lm_head(combined_states)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, **kwargs)
        if "pref_vec" in kwargs:
            inputs["pref_vec"] = kwargs["pref_vec"]
        return inputs