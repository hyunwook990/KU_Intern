import re
import json
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------------
# 1. 설정
# -------------------------------
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)


# -------------------------------
# 2. 데이터 구조
# -------------------------------
@dataclass
class MCQSample:
    question: str
    options: List[str]
    context: Optional[str] = None
    answer: Optional[int] = None  # 정답 인덱스 (0~3)


# -------------------------------
# 3. 모델 로더
# -------------------------------
def load_model(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


# -------------------------------
# 4. KMMLU 로더
# -------------------------------
def normalize_answer_to_index(answer) -> Optional[int]:
    """
    다양한 answer 표현을 0~3 인덱스로 변환
    지원:
    - "A"/"B"/"C"/"D"
    - "a"/"b"/"c"/"d"
    - 0/1/2/3
    - 1/2/3/4
    """
    if answer is None:
        return None

    if isinstance(answer, str):
        ans = answer.strip().upper()
        if ans in {"A", "B", "C", "D"}:
            return ord(ans) - ord("A")
        if ans in {"0", "1", "2", "3"}:
            return int(ans)
        if ans in {"1", "2", "3", "4"}:
            return int(ans) - 1

    if isinstance(answer, int):
        if answer in [0, 1, 2, 3]:
            return answer
        if answer in [1, 2, 3, 4]:
            return answer - 1

    return None


def extract_question(item: dict) -> str:
    for key in ["question", "query", "prompt"]:
        if key in item:
            return str(item[key]).strip()
    raise ValueError(f"질문 필드를 찾을 수 없습니다: {item}")


def extract_context(item: dict) -> Optional[str]:
    for key in ["context", "passage", "article", "paragraph"]:
        if key in item and item[key] is not None and str(item[key]).strip() != "":
            return str(item[key]).strip()
    return None


def extract_options(item: dict) -> List[str]:
    """
    가능한 포맷들:
    1) {"A": "...", "B": "...", "C": "...", "D": "..."}
    2) {"choices": ["...", "...", "...", "..."]}
    3) {"options": ["...", "...", "...", "..."]}
    4) {"choices": {"A": "...", "B": "...", "C": "...", "D": "..."}}
    """
    if all(k in item for k in ["A", "B", "C", "D"]):
        return [
            str(item["A"]),
            str(item["B"]),
            str(item["C"]),
            str(item["D"]),
        ]

    if "choices" in item:
        choices = item["choices"]
        if isinstance(choices, list) and len(choices) == 4:
            return [str(x) for x in choices]
        if isinstance(choices, dict) and all(k in choices for k in ["A", "B", "C", "D"]):
            return [
                str(choices["A"]),
                str(choices["B"]),
                str(choices["C"]),
                str(choices["D"]),
            ]

    if "options" in item:
        options = item["options"]
        if isinstance(options, list) and len(options) == 4:
            return [str(x) for x in options]
        if isinstance(options, dict) and all(k in options for k in ["A", "B", "C", "D"]):
            return [
                str(options["A"]),
                str(options["B"]),
                str(options["C"]),
                str(options["D"]),
            ]

    raise ValueError(f"선택지 필드를 찾을 수 없습니다: {item}")


def extract_answer(item: dict) -> Optional[int]:
    for key in ["answer", "label", "gold", "target"]:
        if key in item:
            return normalize_answer_to_index(item[key])
    return None


def convert_hf_item_to_sample(item: dict) -> MCQSample:
    return MCQSample(
        question=extract_question(item),
        options=extract_options(item),
        context=extract_context(item),
        answer=extract_answer(item)
    )


def load_kmmlu_from_hf(
    dataset_name: str = "HAERAE-HUB/KMMLU",
    subject: str = "Accounting",
    split: str = "test",
    max_samples: Optional[int] = None
) -> List[MCQSample]:
    """
    Hugging Face datasets에서 KMMLU 로드
    예:
        load_kmmlu_from_hf("HAERAE-HUB/KMMLU", "Accounting", "test")
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, subject, split=split)

    samples = []
    for i, item in enumerate(ds):
        try:
            sample = convert_hf_item_to_sample(dict(item))
            samples.append(sample)
        except Exception as e:
            print(f"[Skip] index={i}, reason={e}")

        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples


# -------------------------------
# 5. 프롬프트 구성
# -------------------------------
def format_options(option_indices: List[int], all_options: List[str]) -> str:
    lines = []
    for idx in option_indices:
        label = chr(ord('A') + idx)
        lines.append(f"{label}. {all_options[idx]}")
    return "\n".join(lines)


def build_elimination_prompt(
    question: str,
    options: List[str],
    active_indices: List[int],
    context: Optional[str] = None
) -> str:
    option_text = format_options(active_indices, options)
    context_block = f"Context:\n{context}\n\n" if context else ""

    prompt = f"""
You are solving a multiple-choice question by process of elimination.
Please answer with Korean.

Your task:
1. Read the question carefully.
2. Consider only the currently remaining options.
3. Briefly explain why some options are less plausible.
4. Choose exactly ONE option to eliminate because it is the least plausible.
5. Use the following format exactly:

Reasoning: <your brief reasoning>
Eliminate: <OPTION_LABEL>

Example:
Reasoning: B is not correct because ...
Eliminate: B

{context_block}Question:
{question}

Current remaining options:
{option_text}
""".strip()

    return prompt


# -------------------------------
# 6. 생성 함수
# -------------------------------
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9
) -> str:
    messages = [{"role": "user", "content": prompt}]

    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][input_ids.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()


# -------------------------------
# 7. 제거할 선택지 파싱
# -------------------------------
def parse_elimination_decision(
    response_text: str,
    active_indices: List[int]
) -> Optional[int]:
    valid_labels = {chr(ord('A') + idx): idx for idx in active_indices}

    patterns = [
        r"^\s*Eliminate\s*:\s*([A-Z])\s*$",
        r"Eliminate\s*:\s*([A-Z])",
        r"eliminate\s+option\s+([A-Z])",
        r"remove\s+option\s+([A-Z])",
        r"remove\s+([A-Z])",
        r"eliminate\s+([A-Z])",
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            label = match.group(1).upper()
            if label in valid_labels:
                return valid_labels[label]

    candidates = re.findall(r"\b([A-Z])\b", response_text.upper())
    for c in reversed(candidates):
        if c in valid_labels:
            return valid_labels[c]

    return None


# -------------------------------
# 8. module_1 구현
# -------------------------------
def module_1_eliminate_one(
    model,
    tokenizer,
    question: str,
    context: Optional[str],
    options: List[str],
    current_option_set: List[int],
    verbose: bool = True
) -> Tuple[List[int], str, Optional[int]]:
    prompt = build_elimination_prompt(
        question=question,
        options=options,
        active_indices=current_option_set,
        context=context
    )

    reasoning = generate_text(model, tokenizer, prompt)

    eliminated_idx = parse_elimination_decision(reasoning, current_option_set)

    if eliminated_idx is None:
        raise ValueError(
            f"모델 출력에서 제거할 선택지를 파싱하지 못했습니다.\n\n출력:\n{reasoning}"
        )

    next_option_set = [idx for idx in current_option_set if idx != eliminated_idx]

    if verbose:
        eliminated_label = chr(ord('A') + eliminated_idx)
        print("=" * 60)
        print("[Module 1] Model reasoning:")
        print(reasoning)
        print(f"\n[Module 1] Eliminated option: {eliminated_label}")
        print(f"[Module 1] Remaining options: {[chr(ord('A') + i) for i in next_option_set]}")
        print("=" * 60)

    return next_option_set, reasoning, eliminated_idx


# -------------------------------
# 9. module_2 구현
# -------------------------------
def module_2_iterative_elimination(
    model,
    tokenizer,
    question: str,
    options: List[str],
    context: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    current_option_set = list(range(len(options)))
    t = 0
    history = []

    while len(current_option_set) > 1:
        if verbose:
            print(f"\n[Module 2] Iteration t={t}")

        next_option_set, reasoning, eliminated_idx = module_1_eliminate_one(
            model=model,
            tokenizer=tokenizer,
            question=question,
            context=context,
            options=options,
            current_option_set=current_option_set,
            verbose=verbose
        )

        history.append({
            "step": t,
            "current_option_set": current_option_set.copy(),
            "reasoning": reasoning,
            "eliminated_idx": eliminated_idx,
            "eliminated_label": chr(ord('A') + eliminated_idx),
            "next_option_set": next_option_set.copy()
        })

        current_option_set = next_option_set
        t += 1

    final_idx = current_option_set[0]
    final_label = chr(ord('A') + final_idx)

    result = {
        "predicted_index": final_idx,
        "predicted_label": final_label,
        "predicted_text": options[final_idx],
        "history": history
    }

    if verbose:
        print("\n" + "#" * 60)
        print(f"Final Answer: {final_label}. {options[final_idx]}")
        print("#" * 60)

    return result


# -------------------------------
# 10. 단일 샘플 평가
# -------------------------------
def evaluate_sample(model, tokenizer, sample: MCQSample, verbose: bool = False) -> Dict:
    pred = module_2_iterative_elimination(
        model=model,
        tokenizer=tokenizer,
        question=sample.question,
        options=sample.options,
        context=sample.context,
        verbose=verbose
    )

    predicted_index = pred["predicted_index"]
    is_correct = None if sample.answer is None else (predicted_index == sample.answer)

    return {
        "question": sample.question,
        "options": sample.options,
        "gold_index": sample.answer,
        "gold_label": None if sample.answer is None else chr(ord("A") + sample.answer),
        "predicted_index": predicted_index,
        "predicted_label": pred["predicted_label"],
        "predicted_text": pred["predicted_text"],
        "correct": is_correct,
        "history": pred["history"]
    }


# -------------------------------
# 11. 데이터셋 평가
# -------------------------------
def evaluate_dataset(
    model,
    tokenizer,
    dataset: List[MCQSample],
    verbose: bool = False,
    save_path: Optional[str] = None
) -> Dict:
    results = []
    total = 0
    correct = 0
    skipped = 0

    for idx, sample in enumerate(dataset):
        try:
            result = evaluate_sample(model, tokenizer, sample, verbose=verbose)
            results.append(result)

            if sample.answer is not None:
                total += 1
                correct += int(result["correct"])

            if (idx + 1) % 10 == 0:
                acc = correct / total if total > 0 else 0.0
                print(f"[{idx+1}/{len(dataset)}] accuracy={acc:.4f}, skipped={skipped}")

        except Exception as e:
            skipped += 1
            print(f"[Error] sample index={idx}, reason={e}")

    accuracy = correct / total if total > 0 else None

    output = {
        "accuracy": accuracy,
        "num_evaluated": total,
        "num_correct": correct,
        "num_skipped": skipped,
        "results": results
    }

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    return output


# -------------------------------
# 12. 실행 예시
# -------------------------------
def run_kmmlu_hf_example():
    tokenizer, model = load_model(MODEL_NAME)

    # 예시:
    # subject는 KMMLU의 세부 과목(config name)
    dataset = load_kmmlu_from_hf(
        dataset_name="HAERAE-HUB/KMMLU",
        subject="Accounting",
        split="test",
        max_samples=20   # 처음엔 작게 테스트 추천
    )

    print(f"Loaded samples: {len(dataset)}")

    result = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        verbose=False,
        save_path="kmmlu_accounting_results.json"
    )

    print("\n===== FINAL RESULT =====")
    print("Accuracy:", result["accuracy"])
    print("Num evaluated:", result["num_evaluated"])
    print("Num correct:", result["num_correct"])
    print("Num skipped:", result["num_skipped"])


if __name__ == "__main__":
    run_kmmlu_hf_example()