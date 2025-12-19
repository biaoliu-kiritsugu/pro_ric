import torch
from torch import nn
from transformers import Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

class PrefQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.pref_dim = getattr(config, "pref_dim", 2)
        # 定义 3 层 MLP
        self.pref_mlp = nn.Sequential(
            nn.Linear(self.pref_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, 
                inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, 
                output_hidden_states=None, return_dict=None, pref_vec=None, **kwargs):
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, 
                             past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, 
                             output_attentions=output_attentions, output_hidden_states=output_hidden_states, 
                             return_dict=return_dict, **kwargs)

        hidden_states = outputs[0]
        
        if pref_vec is not None:
            # 类型兼容性处理：如果输入是 list，转换为 tensor
            if not isinstance(pref_vec, torch.Tensor):
                pref_vec = torch.tensor(pref_vec, device=hidden_states.device, dtype=hidden_states.dtype)
            else:
                pref_vec = pref_vec.to(device=hidden_states.device, dtype=hidden_states.dtype)
            
            # 维度补全：如果是 [pref_dim]，转换为 [1, pref_dim] 以支持 batch 操作
            if pref_vec.dim() == 1:
                pref_vec = pref_vec.unsqueeze(0)

            # 映射并相加 (利用广播机制应用到每个 token)
            pref_mapped = self.pref_mlp(pref_vec)  # [batch, hidden_size]
            combined_states = hidden_states + pref_mapped.unsqueeze(1) # [batch, 1, hidden_size]
        else:
            combined_states = hidden_states

        logits = self.lm_head(combined_states)
        
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, **kwargs)
        if "pref_vec" in kwargs:
            inputs["pref_vec"] = kwargs["pref_vec"]
        return inputs