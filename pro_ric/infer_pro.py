import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk


MODEL_DIR = "./pro_summary/best_model"
MAX_LENGTH = 512

dataset = load_from_disk("./datasets/summary_pref1faithfuldeberta_messages.hf")
print(dataset[0])

def _build_user_content(post: str, score_values=None):
    s = "Generate a one-sentence summary of this post: " + post.strip() + " "
    # for i, v in enumerate(score_values):
    #     s += f"<rm{i+1}_score> {v} "
    return s.strip()


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n = int(model.config.num_labels)
    tests = [
        {
            "name": "case_1",
            "post": "A new study finds that drinking coffee in the morning is associated with lower risk of heart disease, but researchers caution correlation is not causation.",
            "scores": [3.0] + [0.1] * (n - 1),
        },
        {
            "name": "case_2",
            "post": "The city council approved a new public transit plan that adds two metro lines and reduces fares for students starting next year.",
            "scores": [0.2] * (n - 1) + [3.0],
        },
        {
            "name": "case_3",
            "post": dataset[100]["messages"][0]["content"],
        }
    ]

    for t in tests:
        user_content = _build_user_content(t["post"])
        messages = [{"role": "user", "content": user_content}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer([text], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits[0]
            probs = F.softmax(logits, dim=-1).detach().cpu()

        print(f"== {t['name']} ==")
        for i in range(n):
            print(f"score{i+1}\t{probs[i].item():.6f}")
        print(f"argmax\tscore{int(probs.argmax().item()) + 1}")
        print()


if __name__ == "__main__":
    main()


