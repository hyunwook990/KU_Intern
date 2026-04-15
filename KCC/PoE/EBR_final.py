import re
import random
import json
import torch
from dataclasses import dataclass
from datasets import load_dataset, get_dataset_config_names
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict


# -------------------------------
# 0. 예외 클래스
# -------------------------------
class EvaluationError(Exception):
    def __init__(
        self,
        message: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        num_calls: int = 0
    ):
        super().__init__(message)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.num_calls = num_calls


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
    subject: Optional[str] = None


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
def convert_item_to_sample(item: dict, subject: Optional[str] = None) -> MCQSample:
    question = str(item["question"]).strip()
    options = [
        str(item["A"]).strip(),
        str(item["B"]).strip(),
        str(item["C"]).strip(),
        str(item["D"]).strip(),
    ]

    answer = item.get("answer", None)

    if answer is not None:
        answer = int(answer)
        if 1 <= answer <= 4:
            answer = answer - 1
        else:
            raise ValueError(f"answer 값이 예상 범위를 벗어났습니다: {answer}")

    return MCQSample(
        question=question,
        options=options,
        answer=answer,
        subject=subject
    )


def get_all_subjects(dataset_name="HAERAE-HUB/KMMLU"):
    return get_dataset_config_names(dataset_name)


def load_kmmlu_dataset_random(
    dataset_name: str,
    subject: str,
    split: str = "test",
    num_samples: int = 100
) -> List[MCQSample]:

    ds = load_dataset(dataset_name, subject, split=split)

    indices = list(range(len(ds)))
    random.shuffle(indices)

    num_samples = min(num_samples, len(ds))
    selected_indices = indices[:num_samples]

    samples = []
    for idx in selected_indices:
        try:
            sample = convert_item_to_sample(dict(ds[idx]), subject=subject)
            samples.append(sample)
        except Exception as e:
            print(f"[Skip] index={idx}, reason={e}")

    return samples


def load_all_subjects_random(
    dataset_name="HAERAE-HUB/KMMLU",
    split="test",
    num_samples_per_subject=100
) -> List[MCQSample]:

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
            print(f"[Skip Subject] {subject}, reason={e}")

    print(f"Loaded all subjects\nTotal subjects: {len(subjects)}")
    return all_samples


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
# 6. 텍스트 생성 + 토큰 수 측정
# -------------------------------
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9
) -> Tuple[str, int, int, int]:
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

    input_tokens = input_ids.shape[-1]

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
    output_tokens = generated.shape[-1]
    total_tokens = input_tokens + output_tokens

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip(), input_tokens, output_tokens, total_tokens


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
) -> Tuple[List[int], str, Optional[int], int, int, int]:
    prompt = build_elimination_prompt(
        question=question,
        options=options,
        active_indices=current_option_set
    )

    reasoning, input_tokens, output_tokens, total_tokens = generate_text(
        model, tokenizer, prompt
    )
    eliminated_idx = parse_elimination_decision(reasoning, current_option_set)

    if eliminated_idx is None:
        raise EvaluationError(
            f"제거할 선택지를 파싱하지 못했습니다.\n\n출력:\n{reasoning}",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            num_calls=1
        )

    next_option_set = [idx for idx in current_option_set if idx != eliminated_idx]

    if verbose:
        eliminated_label = chr(ord("A") + eliminated_idx)
        remaining_labels = [chr(ord("A") + i) for i in next_option_set]

        print("=" * 60)
        print(f"\nEliminated: {eliminated_label}")
        print(f"Remaining: {remaining_labels}")
        print(f"Input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print(f"Total tokens: {total_tokens}")
        print("=" * 60)

    return next_option_set, reasoning, eliminated_idx, input_tokens, output_tokens, total_tokens


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

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    num_calls = 0

    while len(current_option_set) > 1:
        if verbose:
            print(f"\n[Step {step}]")

        try:
            (
                next_option_set,
                reasoning,
                eliminated_idx,
                input_tokens,
                output_tokens,
                step_total_tokens
            ) = eliminate_one_option(
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
                "next_option_set": next_option_set.copy(),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": step_total_tokens
            })

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_tokens += step_total_tokens
            num_calls += 1

            current_option_set = next_option_set
            step += 1

        except EvaluationError as e:
            raise EvaluationError(
                str(e),
                input_tokens=total_input_tokens + e.input_tokens,
                output_tokens=total_output_tokens + e.output_tokens,
                total_tokens=total_tokens + e.total_tokens,
                num_calls=num_calls + e.num_calls
            )

    final_idx = current_option_set[0]

    return {
        "predicted_index": final_idx,
        "predicted_label": chr(ord("A") + final_idx),
        "predicted_text": options[final_idx],
        "history": history,
        "num_calls": num_calls,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens
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
        "subject": sample.subject,
        "question": sample.question,
        "options": sample.options,
        "gold_index": sample.answer,
        "gold_label": None if sample.answer is None else chr(ord("A") + sample.answer),
        "predicted_index": predicted_index,
        "predicted_label": pred["predicted_label"],
        "predicted_text": pred["predicted_text"],
        "correct": is_correct,
        "history": pred["history"],
        "num_calls": pred["num_calls"],
        "total_input_tokens": pred["total_input_tokens"],
        "total_output_tokens": pred["total_output_tokens"],
        "total_tokens": pred["total_tokens"]
    }


# -------------------------------
# 11. 데이터셋 전체 평가
# -------------------------------
def evaluate_dataset(
    model,
    tokenizer,
    dataset: List[MCQSample],
    verbose: bool = False,
    save_path: Optional[str] = None,
) -> Dict:

    subject_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    results = []
    total = 0
    correct = 0
    skipped = 0

    dataset_total_input_tokens = 0
    dataset_total_output_tokens = 0
    dataset_total_tokens = 0
    dataset_total_calls = 0

    for idx, sample in enumerate(dataset):
        try:
            result = evaluate_sample(model, tokenizer, sample, verbose=verbose)
            results.append(result)

            if sample.answer is not None:
                total += 1
                correct += int(result["correct"])
                subject = sample.subject
                subject_stats[subject]["total"] += 1
                subject_stats[subject]["correct"] += int(result["correct"])

            dataset_total_input_tokens += result["total_input_tokens"]
            dataset_total_output_tokens += result["total_output_tokens"]
            dataset_total_tokens += result["total_tokens"]
            dataset_total_calls += result["num_calls"]

            if verbose:
                acc = correct / total if total > 0 else 0.0
                print("subject:", sample.subject)
                print("question:", sample.question)
                print("predicted_label:", result["predicted_label"])
                print("gold:", None if sample.answer is None else chr(ord("A") + sample.answer))
                print(f"sample_num_calls: {result['num_calls']}")
                print(f"sample_total_tokens: {result['total_tokens']}")
                print("=" * 60)
                print(f"[{idx+1}/{len(dataset)}] accuracy={acc:.4f}, skipped={skipped}")

        except EvaluationError as e:
            skipped += 1
            total += 1
            subject_stats[sample.subject]["total"] += 1

            dataset_total_input_tokens += e.input_tokens
            dataset_total_output_tokens += e.output_tokens
            dataset_total_tokens += e.total_tokens
            dataset_total_calls += e.num_calls

            print(f"[Error] sample index={idx}, reason={e}")
            print(
                f"[Partial Usage Added] "
                f"input={e.input_tokens}, output={e.output_tokens}, "
                f"total={e.total_tokens}, calls={e.num_calls}"
            )

        except Exception as e:
            skipped += 1
            total += 1
            subject_stats[sample.subject]["total"] += 1
            print(f"[Error] sample index={idx}, reason={e}")

    # -------------------------------
    # subject별 accuracy 계산
    # -------------------------------
    subject_accuracy = {}
    print("\n===== Subject-wise Accuracy =====")

    for subject, stats in subject_stats.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
        else:
            acc = None

        subject_accuracy[subject] = {
            "accuracy": acc,
            "total": stats["total"],
            "correct": stats["correct"]
        }

        if acc is None:
            print(f"{subject}: accuracy=None, ({stats['correct']}/{stats['total']})")
        else:
            print(f"{subject}: accuracy={acc:.4f}, ({stats['correct']}/{stats['total']})")

    accuracy = correct / total if total > 0 else None
    num_processed = len(results)

    avg_input_tokens = dataset_total_input_tokens / num_processed if num_processed > 0 else None
    avg_output_tokens = dataset_total_output_tokens / num_processed if num_processed > 0 else None
    avg_total_tokens = dataset_total_tokens / num_processed if num_processed > 0 else None
    avg_num_calls = dataset_total_calls / num_processed if num_processed > 0 else None

    print("accuracy:", accuracy)
    print("dataset_total_input_tokens:", dataset_total_input_tokens)
    print("dataset_total_output_tokens:", dataset_total_output_tokens)
    print("dataset_total_tokens:", dataset_total_tokens)
    print("dataset_total_calls:", dataset_total_calls)
    print("avg_input_tokens:", avg_input_tokens)
    print("avg_output_tokens:", avg_output_tokens)
    print("avg_total_tokens:", avg_total_tokens)
    print("avg_num_calls:", avg_num_calls)

    output = {
        "accuracy": accuracy,
        "num_evaluated": total,
        "num_correct": correct,
        "num_skipped": skipped,
        "num_processed": num_processed,
        "dataset_total_input_tokens": dataset_total_input_tokens,
        "dataset_total_output_tokens": dataset_total_output_tokens,
        "dataset_total_tokens": dataset_total_tokens,
        "dataset_total_calls": dataset_total_calls,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "avg_total_tokens": avg_total_tokens,
        "avg_num_calls": avg_num_calls,
        "subject_accuracy": subject_accuracy,
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
    random.seed(42)
    torch.manual_seed(42)
    tokenizer, model = load_model(MODEL_NAME)

    dataset = load_all_subjects_random(
        dataset_name="HAERAE-HUB/KMMLU",
        split="test",
        num_samples_per_subject=50
    )

    print(f"Total samples: {len(dataset)}")

    result = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        verbose=True,
        save_path="EBR_final.json"
    )

    print("\n===== FINAL RESULT =====")
    print("Accuracy:", result["accuracy"])
    print("Num evaluated:", result["num_evaluated"])
    print("Num correct:", result["num_correct"])
    print("Num skipped:", result["num_skipped"])
    print("Num processed:", result["num_processed"])
    print("Dataset total input tokens:", result["dataset_total_input_tokens"])
    print("Dataset total output tokens:", result["dataset_total_output_tokens"])
    print("Dataset total tokens:", result["dataset_total_tokens"])
    print("Dataset total calls:", result["dataset_total_calls"])
    print("Avg input tokens:", result["avg_input_tokens"])
    print("Avg output tokens:", result["avg_output_tokens"])
    print("Avg total tokens:", result["avg_total_tokens"])
    print("Avg num calls:", result["avg_num_calls"])

    print("\n===== Subject-wise Accuracy =====")
    for subject, stats in result["subject_accuracy"].items():
        print(subject, stats)


if __name__ == "__main__":
    main()