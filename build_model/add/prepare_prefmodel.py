import torch
import os
import shutil
from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer
from modeling_pref_qwen3 import PrefQwen3ForCausalLM 

def convert_and_save(local_model_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(f"正在转换模型至 Add 架构...")
    # 1. 配置处理
    config = Qwen3Config.from_pretrained(local_model_path)
    config.pref_dim = 2 
    config.auto_map = {
        "AutoModelForCausalLM": "modeling_pref_qwen3.PrefQwen3ForCausalLM"
    }

    # 2. 实例化新模型
    new_model = PrefQwen3ForCausalLM(config)
    
    # 3. 直接加载原模型权重 (lm_head 会被自动加载，因为名字和形状都对得上)
    # MLP 部分由于原模型没有，会触发 missing_keys，这正是我们想要的
    base_model = Qwen3ForCausalLM.from_pretrained(local_model_path)
    missing_keys, unexpected_keys = new_model.load_state_dict(base_model.state_dict(), strict=False)
    
    print(f"权重迁移完成。新增的 MLP 参数已随机初始化。")
    print(f"Missing keys (应为 mlp 相关): {[k for k in missing_keys if 'pref_mlp' in k]}")

    # 4. 保存
    new_model.save_pretrained(save_path)
    config.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    tokenizer.save_pretrained(save_path)

    # 拷贝定义文件
    shutil.copy("modeling_pref_qwen3.py", os.path.join(save_path, "modeling_pref_qwen3.py"))
    print(f"模型已保存至: {save_path}")

if __name__ == "__main__":
    convert_and_save("/data/xuwenzhe/models/qwen_3_0.6B", "/data/xuwenzhe/models/qwen_3_0.6B_add")