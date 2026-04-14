# 선택지 전부 입력, 하나씩 제거
import re
import json
import torch
from dataclasses import dataclass
from datasets import load_dataset
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------------
# 1. 설정
# -------------------------------
MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)


# -------------------------------
# 2. 데이터 구조
# -------------------------------
@dataclass
class MCQSample:
    question: str
    options: List[str]              # [A, B, C, D]
    answer: Optional[int] = None    # 정답 인덱스: 0~3


# -------------------------------
# 3. 모델 로드
# -------------------------------
def load_model(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


# -------------------------------
# 4. 데이터셋 로드
# -------------------------------
def convert_item_to_sample(item: dict) -> MCQSample:
    question = str(item["question"]).strip()
    options = [
        str(item["A"]).strip(),
        str(item["B"]).strip(),
        str(item["C"]).strip(),
        str(item["D"]).strip(),
    ]

    answer = item.get("answer", None)

    # answer가 정수라고 가정
    if answer is not None:
        answer = int(answer)
        if 1 <= answer <= 4:
            answer = answer - 1
        else:
            raise ValueError(f"answer 값이 예상 범위를 벗어났습니다: {answer}")

    return MCQSample(
        question=question,
        options=options,
        answer=answer
    )


def load_kmmlu_dataset(
    dataset_name: str = "HAERAE-HUB/KMMLU",
    subject: str = "Accounting",
    split: str = "test",
    max_samples: Optional[int] = None
) -> List[MCQSample]:
    ds = load_dataset(dataset_name, subject, split=split)

    samples = []
    for i, item in enumerate(ds):
        try:
            sample = convert_item_to_sample(dict(item))
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
        label = chr(ord("A") + idx)
        lines.append(f"{label}. {all_options[idx]}")
    return "\n".join(lines)


def build_elimination_prompt(
    question: str,
    options: List[str],
    active_indices: List[int]
) -> str:
    option_text = format_options(active_indices, options)
    
    ko_prompt = f"""
당신은 객관식 문제를 "제거 방식"으로 푸는 전문가입니다.

작업:
1. 문제를 주의 깊게 읽으세요.
2. 현재 남아 있는 선택지만 고려하세요.
3. 각 선택지 중 덜 타당한 것을 간단히 설명하세요.
4. 가장 가능성이 낮은 선택지 하나를 반드시 제거하세요.
5. 아래 형식을 정확히 따르세요.

형식:
Reasoning: <간단한 설명>
Eliminate: <선택지 라벨>

예시:
Reasoning: B는 문제 조건과 맞지 않으므로 가장 가능성이 낮습니다.
Eliminate: B

문제:
{question}

현재 남은 선택지:
{option_text}

중요:
- 반드시 하나의 선택지만 제거하세요.
- Reasoning과 Eliminate 형식을 반드시 지키세요.
- "Eliminate" 키워드는 영어 그대로 사용하세요.
- 다른 텍스트는 추가하지 마세요.
""".strip()

    prompt = f"""
You are solving a multiple-choice question by process of elimination.
Please answer in Korean.

Your task:
1. Read the question carefully.
2. Consider only the currently remaining options.
3. Briefly explain why some options are less plausible.
4. Choose exactly ONE option to eliminate because it is the least plausible.
5. Use the following format exactly:

Reasoning: <your brief reasoning>
Eliminate: <OPTION_LABEL>

Example:
Reasoning: B는 문제 조건에 맞지 않으므로 가장 가능성이 낮습니다.
Eliminate: B

Question:
{question}

Current remaining options:
{option_text}
""".strip()

    return prompt


# -------------------------------
# 6. 텍스트 생성
# -------------------------------
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
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
# 7. 제거 선택지 파싱
# -------------------------------
def parse_elimination_decision(
    response_text: str,
    active_indices: List[int]
) -> Optional[int]:
    valid_labels = {chr(ord("A") + idx): idx for idx in active_indices}

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

    # 혹시 형식이 조금 어긋나면 마지막 등장 알파벳을 사용
    candidates = re.findall(r"\b([A-Z])\b", response_text.upper())
    for c in reversed(candidates):
        if c in valid_labels:
            return valid_labels[c]

    return None


# -------------------------------
# 8. 한 번에 선택지 1개 제거
# -------------------------------
def eliminate_one_option(
    model,
    tokenizer,
    question: str,
    options: List[str],
    current_option_set: List[int],
    verbose: bool = True
) -> Tuple[List[int], str, Optional[int]]:
    prompt = build_elimination_prompt(
        question=question,
        options=options,
        active_indices=current_option_set
    )

    reasoning = generate_text(model, tokenizer, prompt)
    eliminated_idx = parse_elimination_decision(reasoning, current_option_set)

    if eliminated_idx is None:
        raise ValueError(f"제거할 선택지를 파싱하지 못했습니다.\n\n출력:\n{reasoning}")

    next_option_set = [idx for idx in current_option_set if idx != eliminated_idx]

    if verbose:
        eliminated_label = chr(ord("A") + eliminated_idx)
        remaining_labels = [chr(ord("A") + i) for i in next_option_set]

        print("=" * 60)
        # print("[Model reasoning]")
        # print(reasoning)
        print(f"\nEliminated: {eliminated_label}")
        print(f"Remaining: {remaining_labels}")
        print("=" * 60)

    return next_option_set, reasoning, eliminated_idx


# -------------------------------
# 9. 반복 제거로 최종 답 선택
# -------------------------------
def solve_by_iterative_elimination(
    model,
    tokenizer,
    question: str,
    options: List[str],
    verbose: bool = True
) -> Dict:
    current_option_set = [0, 1, 2, 3]
    history = []
    step = 0

    while len(current_option_set) > 1:
        if verbose:
            print(f"\n[Step {step}]")

        next_option_set, reasoning, eliminated_idx = eliminate_one_option(
            model=model,
            tokenizer=tokenizer,
            question=question,
            options=options,
            current_option_set=current_option_set,
            verbose=verbose
        )

        history.append({
            "step": step,
            "current_option_set": current_option_set.copy(),
            "reasoning": reasoning,
            "eliminated_idx": eliminated_idx,
            "eliminated_label": chr(ord("A") + eliminated_idx),
            "next_option_set": next_option_set.copy()
        })

        current_option_set = next_option_set
        step += 1

    final_idx = current_option_set[0]

    return {
        "predicted_index": final_idx,
        "predicted_label": chr(ord("A") + final_idx),
        "predicted_text": options[final_idx],
        "history": history
    }


# -------------------------------
# 10. 단일 샘플 평가
# -------------------------------
def evaluate_sample(model, tokenizer, sample: MCQSample, verbose: bool = False) -> Dict:
    pred = solve_by_iterative_elimination(
        model=model,
        tokenizer=tokenizer,
        question=sample.question,
        options=sample.options,
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
# 11. 데이터셋 전체 평가
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

            if verbose:
                acc = correct / total if total > 0 else 0.0
                print("predicted_label:", result["predicted_label"])
                print("gold:",chr(ord("A")+sample.answer))
                print("="*60)
                print(f"[{idx+1}/{len(dataset)}] accuracy={acc:.4f}, skipped={skipped}")

        except Exception as e:
            skipped += 1
            print(f"[Error] sample index={idx}, reason={e}")

    accuracy = correct / total if total > 0 else None
    print("accuracy:", accuracy)

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
def main():
    tokenizer, model = load_model(MODEL_NAME)

    dataset = load_kmmlu_dataset(
        dataset_name="HAERAE-HUB/KMMLU",
        subject="Accounting",
        split="test",
        max_samples=100
    )

    print(f"Loaded samples: {len(dataset)}")

    result = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        verbose=True,
        save_path="ebr_accounting_test.json"
    )

    print("\n===== FINAL RESULT =====")
    print("Accuracy:", result["accuracy"])
    print("Num evaluated:", result["num_evaluated"])
    print("Num correct:", result["num_correct"])
    print("Num skipped:", result["num_skipped"])


if __name__ == "__main__":
    main()