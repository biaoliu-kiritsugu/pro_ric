import torch
import os
import shutil
from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer
# 导入我们刚才写的类用于初始化
from modeling_pref_qwen3 import PrefQwen3ForCausalLM 

def convert_and_save(local_model_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(f"正在加载原模型...")
    config = Qwen3Config.from_pretrained(local_model_path)
    config.pref_dim = 2 
    config.tie_word_embeddings = False
    # 核心配置：添加 auto_map 使得 AutoModel 能找到自定义类
    config.auto_map = {
        "AutoModelForCausalLM": "modeling_pref_qwen3.PrefQwen3ForCausalLM"
    }

    base_model = Qwen3ForCausalLM.from_pretrained(local_model_path)
    new_model = PrefQwen3ForCausalLM(config)

    # 修复报错：过滤掉 lm_head.weight
    state_dict = base_model.state_dict()
    old_lm_head_weight = state_dict.pop("lm_head.weight") # 弹出形状不匹配的权重
    
    # 加载剩余权重（此时 strict=True 也不会报错了）
    new_model.load_state_dict(state_dict, strict=False)

    # 手动迁移并扩展 LM Head 权重
    with torch.no_grad():
        new_model.lm_head.weight[:, :config.hidden_size].copy_(old_lm_head_weight)
        # 其余 2 列保持随机初始化
        print("成功扩展 LM Head 权重。")

    # 保存模型、配置和分词器
    new_model.save_pretrained(save_path)
    config.save_pretrained(save_path) # 显式保存带 auto_map 的 config
    
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    tokenizer.save_pretrained(save_path)

    # 重要：将模型定义文件拷贝到目标文件夹
    shutil.copy("modeling_pref_qwen3.py", os.path.join(save_path, "modeling_pref_qwen3.py"))
    print(f"模型已转换并存至: {save_path}")
# 执行转换 (请替换为你本地的路径)
convert_and_save("/data/xuwenzhe/models/qwen_3_0.6B", "/data/xuwenzhe/qwen_3_0.6B_concat")