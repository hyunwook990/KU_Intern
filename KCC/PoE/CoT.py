import json
import re
import random
from dataclasses import dataclass
from typing import List, Optional, Dict

import torch
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict


# =========================================================
# 설정
# =========================================================

MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)


# =========================================================
# Data Classes
# =========================================================

@dataclass
class QuestionSample:
    question: str
    options: List[str]
    answer: Optional[str] = None
    subject: Optional[str] = None


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


# =========================================================
# Prompt
# =========================================================

DIRECT_COT_PROMPT = """당신은 객관식 문제를 푸는 전문가입니다.

문제:
{question}

선택지:
{options_text}

지시:
1. 문제를 차근차근 생각하세요.
2. 각 선택지를 비교하여 가장 적절한 답을 고르세요.

반드시 아래 JSON 형식으로만 답하세요:
{{
  "reasoning": "간단한 설명",
  "answer": "A"
}}

규칙:
- answer에는 반드시 하나의 선택지 라벨만 넣으세요.
- JSON 이외의 텍스트는 출력하지 마세요.
"""


# =========================================================
# LLM Wrapper
# =========================================================

class HFLLM:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device_map: str = "auto",
        dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
        max_new_tokens: int = 512,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map if torch.cuda.is_available() else None,
        )

    def generate(self, prompt: str, temperature: float = 0.0) -> GenerationResult:
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.model.device)

        input_tokens = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        output_tokens = new_tokens.shape[0]
        total_tokens = input_tokens + output_tokens
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens
        )


# =========================================================
# Utils
# =========================================================

def option_label(index: int) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return labels[index]


def answer_to_label(answer) -> Optional[str]:
    if answer is None:
        return None
    answer = int(answer)
    return option_label(answer - 1)


def extract_json_block(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"JSON block not found:\n{text}")
    return match.group(0)


def safe_json_loads(text: str) -> dict:
    raw = extract_json_block(text)
    raw = raw.replace("“", '"').replace("”", '"').replace("’", "'")
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)
    return json.loads(raw)


def convert_item_to_sample(item: dict, subject: Optional[str] = None) -> QuestionSample:
    options = []
    for key in ["A", "B", "C", "D"]:
        if key in item:
            options.append(str(item[key]).strip())

    return QuestionSample(
        question=str(item["question"]).strip(),
        options=options,
        answer=answer_to_label(item.get("answer")),
        subject=subject
    )


def get_all_subjects(dataset_name: str = "HAERAE-HUB/KMMLU") -> List[str]:
    return get_dataset_config_names(dataset_name)


def load_kmmlu_dataset_random(
    dataset_name: str = "HAERAE-HUB/KMMLU",
    subject: str = "Accounting",
    split: str = "test",
    num_samples: int = 100
) -> List[QuestionSample]:
    ds = load_dataset(dataset_name, subject, split=split)

    indices = list(range(len(ds)))
    random.shuffle(indices)
    selected_indices = indices[:min(num_samples, len(ds))]

    samples = []
    for idx in selected_indices:
        try:
            samples.append(convert_item_to_sample(dict(ds[idx]), subject=subject))
        except Exception as e:
            print(f"[Skip] subject={subject}, index={idx}, reason={e}")

    return samples


def load_all_subjects_random(
    dataset_name: str = "HAERAE-HUB/KMMLU",
    split: str = "test",
    num_samples_per_subject: int = 100
) -> List[QuestionSample]:
    subjects = get_all_subjects(dataset_name)
    all_samples = []

    for subject in subjects:
        try:
            samples = load_kmmlu_dataset_random(
                dataset_name=dataset_name,
                subject=subject,
                split=split,
                num_samples=num_samples_per_subject
            )
            all_samples.extend(samples)
        except Exception as e:
            print(f"[Skip Subject] subject={subject}, reason={e}")

    print(f"Loaded total samples: {len(all_samples)}")
    return all_samples


def build_options_text(options: List[str]) -> str:
    lines = []
    for i, opt in enumerate(options):
        lines.append(f"{option_label(i)}. {opt}")
    return "\n".join(lines)


def build_direct_cot_prompt(sample: QuestionSample) -> str:
    return DIRECT_COT_PROMPT.format(
        question=sample.question,
        options_text=build_options_text(sample.options)
    )


def parse_direct_cot_output(response_text: str, num_options: int) -> Optional[Dict]:
    valid_labels = [option_label(i) for i in range(num_options)]

    try:
        data = safe_json_loads(response_text)
    except Exception:
        return None

    reasoning = str(data.get("reasoning", "")).strip()

    if "answer" not in data:
        return None

    answer = str(data["answer"]).strip().upper()
    if answer not in valid_labels:
        return None

    return {
        "reasoning": reasoning,
        "answer": answer
    }


# =========================================================
# Direct CoT Inference
# =========================================================

def solve_direct_cot(
    llm: HFLLM,
    sample: QuestionSample,
    temperature: float = 0.0
) -> Dict:
    prompt = build_direct_cot_prompt(sample)
    gen_result = llm.generate(prompt, temperature=temperature)

    parsed = parse_direct_cot_output(gen_result.text, len(sample.options))

    if parsed is None:
        pred_label = None
        reasoning = None
    else:
        pred_label = parsed["answer"]
        reasoning = parsed["reasoning"]

    return {
        "prediction": pred_label,
        "reasoning": reasoning,
        "raw_output": gen_result.text,
        "input_tokens": gen_result.input_tokens,
        "output_tokens": gen_result.output_tokens,
        "total_tokens": gen_result.total_tokens,
    }


# =========================================================
# Evaluation
# =========================================================

def evaluate_direct_cot(
    llm: HFLLM,
    dataset: List[QuestionSample],
    temperature: float = 0.0,
    verbose: bool = True
) -> Dict:
    predictions = []
    correct = 0
    total = 0
    skipped = 0

    subject_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    for idx, sample in enumerate(dataset):
        try:
            output = solve_direct_cot(llm, sample, temperature=temperature)
            pred_label = output["prediction"]

            # parse 실패(None)도 total에 포함되므로 실질적으로 오답 처리
            is_correct = None if sample.answer is None or pred_label is None else (pred_label == sample.answer)

            row = {
                "subject": sample.subject,
                "question": sample.question,
                "options": sample.options,
                "prediction": pred_label,
                "gold": sample.answer,
                "correct": is_correct,
                "reasoning": output["reasoning"],
                "raw_output": output["raw_output"],
                "input_tokens": output["input_tokens"],
                "output_tokens": output["output_tokens"],
                "total_tokens": output["total_tokens"],
            }
            predictions.append(row)

            total += 1
            subject_stats[sample.subject]["total"] += 1

            if is_correct is True:
                correct += 1
                subject_stats[sample.subject]["correct"] += 1

            if pred_label is None:
                skipped += 1

            total_input_tokens += output["input_tokens"]
            total_output_tokens += output["output_tokens"]
            total_tokens += output["total_tokens"]

            if verbose:
                acc_so_far = correct / total if total > 0 else 0.0
                print("====================================================")
                print(output["reasoning"])
                print("====================================================")
                print(f"[{idx+1}/{len(dataset)}]")
                print("subject:", sample.subject)
                print("prediction:", pred_label)
                print("gold:", sample.answer)
                print("correct:", is_correct)
                print("input_tokens:", output["input_tokens"])
                print("output_tokens:", output["output_tokens"])
                print("total_tokens:", output["output_tokens"]+output["input_tokens"])
                print("current_acc:", round(acc_so_far, 4))
                print("====================================================")

        except Exception as e:
            skipped += 1
            total += 1
            subject_stats[sample.subject]["total"] += 1
            print(f"[Error] sample index={idx}, reason={e}")

    subject_accuracy = {}
    for subject, stats in subject_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else None
        subject_accuracy[subject] = {
            "accuracy": acc,
            "total": stats["total"],
            "correct": stats["correct"]
        }

    final_acc = correct / total if total > 0 else None

    return {
        "accuracy": final_acc,
        "num_evaluated": total,
        "num_correct": correct,
        "num_skipped": skipped,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "avg_input_tokens": total_input_tokens / total if total > 0 else None,
        "avg_output_tokens": total_output_tokens / total if total > 0 else None,
        "avg_total_tokens": total_tokens / total if total > 0 else None,
        "subject_accuracy": subject_accuracy,
        "results": predictions
    }


def save_results_json(result: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    llm = HFLLM(
        model_name=MODEL_NAME,
        device_map="auto",
        max_new_tokens=512
    )

    dataset = load_all_subjects_random(
        dataset_name="HAERAE-HUB/KMMLU",
        split="test",
        num_samples_per_subject=1
    )

    result = evaluate_direct_cot(
        llm=llm,
        dataset=dataset,
        temperature=0.0,
        verbose=True
    )

    print("\n===== Final Result =====")
    print("Accuracy:", result["accuracy"])
    print("Num evaluated:", result["num_evaluated"])
    print("Num correct:", result["num_correct"])
    print("Num skipped:", result["num_skipped"])
    print("total_input_tokens", result["total_input_tokens"])
    print("total_output_tokens", result["total_ouput_tokens"])
    print("total_output_tokens", result["total_tokens"])
    print("Avg input tokens:", result["avg_input_tokens"])
    print("Avg output tokens:", result["avg_output_tokens"])
    print("Avg total tokens:", result["avg_total_tokens"])

    save_results_json(result, "direct_cot_baseline.json")