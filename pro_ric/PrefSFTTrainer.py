import torch
import torch.nn as nn
from trl import SFTTrainer
from transformers import Trainer
from typing import Optional, Dict, Any, Union
import torch.nn.functional as F
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Dict, List, Optional, Tuple, Any
from trl import DataCollatorForCompletionOnlyLM
class PrefSFTTrainer(SFTTrainer):
    def __init__(self, score_classifier: Optional[nn.Module] = None,pref_learning_rate=1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_classifier = score_classifier
        
        if self.score_classifier is not None:
            # 将分类器移动到主模型相同的设备和精度
            self.score_classifier.to(self.model.device)
            if hasattr(self.model, "dtype"):
                self.score_classifier.to(self.model.dtype)
            
            # 确保分类器在训练模式
            self.score_classifier.train()
        self.pref_learning_rate=pref_learning_rate
    def create_optimizer(self):
        """
        重写优化器创建逻辑，将 score_classifier 的参数加入优化器，
        这样 SFT loss 的梯度才能反向传播并更新它。
        """
        if self.optimizer is None:
            # 获取主模型和分类器的参数
            # 使用 getattr 处理可能被包装过的模型
            m = self.model.module if hasattr(self.model, "module") else self.model
            c = self.score_classifier.module if hasattr(self.score_classifier, "module") else self.score_classifier
            
            model_params = filter(lambda p: p.requires_grad, m.parameters())
            params = [{"params": model_params}]
            
            if c is not None:
                classifier_params = filter(lambda p: p.requires_grad, c.parameters())
                params.append({"params": classifier_params, "lr": self.pref_learning_rate})
            
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(params, **optimizer_kwargs)
            
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写损失计算逻辑。
        """
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        target_pref_vec = inputs.get("pref_vec")
        if isinstance(target_pref_vec, list):
            # 检查列表里的元素是不是 Tensor
            if len(target_pref_vec) > 0 and torch.is_tensor(target_pref_vec[0]):
                # 如果是 Tensor 列表，使用 stack 堆叠成 [Batch, 2]
                target_pref_vec = torch.stack(target_pref_vec)
            else:
                # 如果是普通的 float 列表，才用 torch.tensor
                target_pref_vec = torch.tensor(target_pref_vec)
        # 1. 获取 pref_vec
        if self.score_classifier is not None:
            # 分类器输出，假设未经过 Softmax (logits 空间)
            pred_pref_logits = self.score_classifier(
                inputs.get("prompt"),
            )
            # 用于传给主模型进行拼接的向量 (保持梯度)
            # 我们假设主模型预期的是概率分布或激活后的值
            pred_pref_vec = pred_pref_logits
        else:
            pred_pref_vec = target_pref_vec
        # 2. 将 pref_vec 传入主模型的 forward
        # 我们之前改写的 PrefQwen3ForCausalLM 接收 pref_vec 参数
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels,
            pref_vec=pred_pref_vec,
            return_dict=True
        )
        
        # 3. 获取损失
        # Trainer 默认从 outputs.loss 获取损失
        sft_loss = outputs.loss
        total_loss = sft_loss
        if self.score_classifier is not None and target_pref_vec is not None:
            # 计算 KL 散度: KL(Target || Pred) 或 KL(Pred || Target)
            # PyTorch 的 kl_div 预期输入是 log_probs
            log_probs = F.log_softmax(pred_pref_logits, dim=-1)
            
            # 如果 target_pref_vec 已经是分布，则直接使用
            # 如果是 hard label，需要转换成概率分布
            if target_pref_vec.sum(dim=-1)[0] != 1.0: # 简单判断是否归一化
                target_dist = F.softmax(target_pref_vec, dim=-1)
            else:
                target_dist = target_pref_vec
            target_dist = target_dist.to(log_probs.device)
            kl_loss = F.kl_div(log_probs, target_dist, reduction="batchmean")
            
            # 合并损失
            total_loss = sft_loss +  kl_loss

        return (total_loss, outputs) if return_outputs else total_loss
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
            # 保存主模型
            super().save_model(output_dir, _internal_call)
            
            # 保存偏好模型
            if self.score_classifier is not None and self.accelerator.is_main_process:
                # 确保解包后再保存，避免保存 DDP wrapper
                unwrapped_classifier = self.accelerator.unwrap_model(self.score_classifier)
                state_dict = unwrapped_classifier.state_dict()
                torch.save(state_dict, f"{output_dir}/score_classifier.pt")
class PrefDataCollator(DataCollatorForCompletionOnlyLM):
    def __init__(self, tokenizer, **kwargs):
        # 1. 自动探测 response_template
        response_template = "</think>\n\n"
        print(f"--- Detected response_template: {repr(response_template)} ---")
        
        # 2. 调用父类初始化
        super().__init__(response_template=response_template, tokenizer=tokenizer, **kwargs)
        
        # 3. 定义数值和张量白名单
        self.whitelist = ["input_ids", "attention_mask", "labels"]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1. Pop 出你的自定义字段（注意：features 是 List[Dict]）
        # 这里直接通过列表推导式提取，不留在 features 里
        prompts = [f.pop("prompt", "") for f in features]
        pref_vecs = [f.pop("pref_vec", None) for f in features]
        
        # 2. 白名单过滤：清理掉数据集里多余的列（如 raw_text, id 等）
        # 只有在 whitelist 里的键会被保留传给父类进行 Padding
        cleaned_features = [
            {k: v for k, v in f.items() if k in self.whitelist}
            for f in features
        ]
        
        # 3. 调用父类：执行自动分词对齐和 Labels 生成 (Masking)
        # 此时 cleaned_features 只包含标准的 input_ids 等
        batch = super().__call__(cleaned_features)
        
        # 4. 手动将字段塞回 batch
        batch["prompt"] = prompts
        batch["pref_vec"] = pref_vecs
            
        return batch