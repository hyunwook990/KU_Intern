import json
import re
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

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


@dataclass
class DirectRationaleResult:
    rationale: str
    answer: str
    raw_output: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    num_calls: int = 1


class DirectRationaleExecutionError(Exception):
    def __init__(self, message: str, usage: Dict, raw_output: Optional[str] = None):
        super().__init__(message)
        self.usage = usage
        self.raw_output = raw_output


# =========================================================
# Prompt
# =========================================================

DIRECT_RATIONALE_PROMPT = """당신은 객관식 문제를 푸는 전문가입니다.

문제:
{question}

선택지:
{options_text}

작업:
1. 각 선택지를 간단히 비교 검토하세요.
2. 그 비교를 바탕으로 최종 정답 하나를 고르세요.
3. 최종 출력에는 비교 결과를 요약한 rationale만 쓰세요.

반드시 아래 JSON 형식으로만 답하세요:
{{
  "rationale": "선택지 비교를 바탕으로 한 간단한 최종 판단 근거",
  "answer": "A"
}}

규칙:
- rationale은 비어 있으면 안 됩니다.
- rationale에는 선택지 간 비교 판단이 드러나야 합니다.
- rationale은 간단히 쓰세요.
- answer에는 반드시 하나의 선택지 라벨만 넣으세요.
- answer에는 선택지 내용 전체나 일부를 쓰지 말고 라벨만 쓰세요.
- JSON 이외의 텍스트는 출력하지 마세요.
"""


REPAIR_DIRECT_RATIONALE_PROMPT = """아래 출력은 형식이 깨졌거나 JSON 파싱이 실패했습니다.
의미는 최대한 유지하고 반드시 JSON만 다시 출력하세요.

원래 출력:
{raw_output}

반드시 아래 형식으로만 출력하세요:
{{
  "rationale": "선택지 비교를 바탕으로 한 간단한 최종 판단 근거",
  "answer": "A"
}}

규칙:
- JSON 이외의 텍스트는 절대 출력하지 마세요.
- rationale은 비어 있으면 안 됩니다.
- answer는 반드시 하나의 라벨만 출력하세요. 예: "A"
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
# Parsing Utils
# =========================================================

def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def normalize_quotes(text: str) -> str:
    return (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("＂", '"')
        .replace("＇", "'")
    )


def extract_json_candidates(text: str) -> List[str]:
    text = strip_code_fences(normalize_quotes(text))
    candidates = []
    stack = []
    start = None

    for i, ch in enumerate(text):
        if ch == "{":
            if not stack:
                start = i
            stack.append(ch)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(text[start:i + 1])
                    start = None

    if text.startswith("{") and text.endswith("}"):
        candidates.append(text)

    unique = []
    seen = set()
    for c in candidates:
        c = c.strip()
        if c and c not in seen:
            unique.append(c)
            seen.add(c)
    return unique


def escape_invalid_backslashes_in_json_string(raw: str) -> str:
    result = []
    in_string = False
    i = 0
    n = len(raw)

    while i < n:
        ch = raw[i]

        if ch == '"':
            backslash_count = 0
            j = i - 1
            while j >= 0 and raw[j] == "\\":
                backslash_count += 1
                j -= 1
            if backslash_count % 2 == 0:
                in_string = not in_string
            result.append(ch)
            i += 1
            continue

        if ch == "\\" and in_string:
            if i + 1 >= n:
                result.append("\\\\")
                i += 1
                continue

            nxt = raw[i + 1]

            if nxt in ['"', "\\", "/", "b", "f", "n", "r", "t"]:
                result.append("\\")
                result.append(nxt)
                i += 2
                continue

            if nxt == "u":
                hex_part = raw[i + 2:i + 6]
                if len(hex_part) == 4 and re.fullmatch(r"[0-9a-fA-F]{4}", hex_part):
                    result.append("\\u")
                    result.append(hex_part)
                    i += 6
                    continue

            result.append("\\\\")
            result.append(nxt)
            i += 2
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def clean_json_like_string(raw: str) -> str:
    raw = raw.strip()
    raw = normalize_quotes(raw)
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)
    raw = re.sub(r"(?<=\{|,)\s*'([^']+)'\s*:", r' "\1":', raw)
    raw = re.sub(
        r':\s*\'([^\']*)\'',
        lambda m: ': "' + m.group(1).replace('"', '\\"') + '"',
        raw
    )
    raw = escape_invalid_backslashes_in_json_string(raw)
    return raw


def try_json_loads_variants(raw: str) -> dict:
    variants = [raw, clean_json_like_string(raw)]
    last_err = None

    for candidate in variants:
        try:
            return json.loads(candidate)
        except Exception as e:
            last_err = e

    raise last_err


def safe_json_loads(text: str) -> dict:
    candidates = extract_json_candidates(text)
    if not candidates:
        raise ValueError(f"JSON block not found:\n{text}")

    parsed_objects = []
    for candidate in candidates:
        try:
            parsed_objects.append(try_json_loads_variants(candidate))
        except Exception:
            continue

    if not parsed_objects:
        raise ValueError(f"Valid JSON block not found:\n{text}")

    for obj in reversed(parsed_objects):
        if isinstance(obj, dict):
            return obj

    raise ValueError(f"Parsed JSON exists but no dict found:\n{text}")


def canonicalize_keys(data: dict) -> dict:
    key_aliases = {
        "rationale": "rationale",
        "reasoning": "rationale",
        "reason": "rationale",
        "explanation": "rationale",
        "answer": "answer",
        "final_answer": "answer",
    }

    normalized = {}
    for k, v in data.items():
        nk = key_aliases.get(str(k).strip().lower(), str(k).strip().lower())
        normalized[nk] = v
    return normalized


def option_label(index: int) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(labels):
        return labels[index]
    return f"Option{index + 1}"


def answer_to_label(answer) -> Optional[str]:
    if answer is None:
        return None

    answer = int(answer)
    if 1 <= answer <= 26:
        return option_label(answer - 1)

    raise ValueError(f"지원하지 않는 answer 값: {answer}")


def label_to_index(label: str) -> int:
    label = str(label).strip().upper()
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if label not in labels:
        raise ValueError(f"Invalid label: {label}")
    return labels.index(label)


def normalize_label(raw_label: str, valid_labels: Optional[List[str]] = None) -> str:
    s = str(raw_label).strip().upper()

    patterns = [
        r"^([A-Z])$",
        r"^([A-Z])[\.\)]?$",
        r"^OPTION\s*([A-Z])$",
        r"^ANSWER\s*[:\-]?\s*([A-Z])$",
        r"^정답\s*[:\-]?\s*([A-Z])$",
    ]

    for pattern in patterns:
        m = re.match(pattern, s)
        if m:
            candidate = m.group(1)
            if valid_labels is None or candidate in valid_labels:
                return candidate

    s_alnum = re.sub(r"[^A-Z0-9가-힣]", "", s)

    if s_alnum.isdigit():
        idx = int(s_alnum) - 1
        labels = valid_labels if valid_labels is not None else list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if 0 <= idx < len(labels):
            return labels[idx]

    m = re.search(r"\b([A-Z])\b", s)
    if m:
        candidate = m.group(1)
        if valid_labels is None or candidate in valid_labels:
            return candidate

    raise ValueError(f"Invalid label text: {raw_label}")


def normalize_option_text_for_match(text: str) -> str:
    text = str(text).strip().lower()
    text = normalize_quotes(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'“”‘’`]", "", text)
    text = re.sub(r"[.,;:!?()\[\]{}<>]", "", text)
    return text.strip()


def resolve_answer_to_label(
    raw_answer,
    valid_labels: List[str],
    option_texts: List[str]
) -> str:
    try:
        return normalize_label(str(raw_answer), valid_labels=valid_labels)
    except Exception:
        pass

    answer_text = str(raw_answer).strip()
    norm_answer = normalize_option_text_for_match(answer_text)

    for label, text in zip(valid_labels, option_texts):
        if answer_text == str(text).strip():
            return label

    normalized_text_map = {
        label: normalize_option_text_for_match(text)
        for label, text in zip(valid_labels, option_texts)
    }

    for label, norm_text in normalized_text_map.items():
        if norm_answer == norm_text:
            return label

    for label, text in zip(valid_labels, option_texts):
        merged_candidates = [
            f"{label}. {text}",
            f"{label}) {text}",
            f"{label} {text}",
            f"정답: {label}. {text}",
            f"answer: {label}. {text}",
        ]
        merged_candidates = [normalize_option_text_for_match(x) for x in merged_candidates]
        if norm_answer in merged_candidates:
            return label

    contains_matches = []
    for label, norm_text in normalized_text_map.items():
        if norm_answer and (norm_answer in norm_text or norm_text in norm_answer):
            contains_matches.append(label)

    if len(contains_matches) == 1:
        return contains_matches[0]

    raise ValueError(f"Invalid answer text: {raw_answer}")


def repair_direct_rationale_output(
    llm: HFLLM,
    raw_output: str,
    temperature: float = 0.0
) -> GenerationResult:
    prompt = REPAIR_DIRECT_RATIONALE_PROMPT.format(raw_output=raw_output)
    return llm.generate(prompt, temperature=temperature)


def parse_direct_rationale_output_from_text(
    raw_output: str,
    option_texts: List[str],
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    num_calls: int = 1
) -> DirectRationaleResult:
    valid_labels = [option_label(i) for i in range(len(option_texts))]
    data = canonicalize_keys(safe_json_loads(raw_output))

    rationale = str(data.get("rationale", "")).strip()
    raw_answer = data.get("answer", None)

    if not rationale:
        raise ValueError(f"Empty rationale:\n{raw_output}")

    answer = resolve_answer_to_label(
        raw_answer=raw_answer,
        valid_labels=valid_labels,
        option_texts=option_texts
    )

    return DirectRationaleResult(
        rationale=rationale,
        answer=answer,
        raw_output=raw_output,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        num_calls=num_calls
    )


def parse_direct_rationale_output(
    llm: HFLLM,
    raw_output: str,
    option_texts: List[str],
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    temperature: float = 0.0
) -> DirectRationaleResult:
    try:
        return parse_direct_rationale_output_from_text(
            raw_output=raw_output,
            option_texts=option_texts,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            num_calls=1
        )
    except Exception:
        repaired = repair_direct_rationale_output(llm, raw_output, temperature=temperature)
        return parse_direct_rationale_output_from_text(
            raw_output=repaired.text,
            option_texts=option_texts,
            input_tokens=input_tokens + repaired.input_tokens,
            output_tokens=output_tokens + repaired.output_tokens,
            total_tokens=total_tokens + repaired.total_tokens,
            num_calls=2
        )


# =========================================================
# Dataset Utils
# =========================================================

def convert_item_to_sample(item: dict, subject: Optional[str] = None) -> QuestionSample:
    options = []
    for key in ["A", "B", "C", "D"]:
        if key in item:
            options.append(str(item[key]).strip())

    if not options:
        raise ValueError("선택지를 찾을 수 없습니다.")

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


# =========================================================
# Prompt Builder
# =========================================================

def build_options_text(options: List[str]) -> str:
    lines = []
    for i, opt in enumerate(options):
        lines.append(f"{option_label(i)}. {opt}")
    return "\n".join(lines)


def build_direct_rationale_prompt(sample: QuestionSample) -> str:
    return DIRECT_RATIONALE_PROMPT.format(
        question=sample.question,
        options_text=build_options_text(sample.options)
    )


# =========================================================
# Direct Comparative Rationale Inference
# =========================================================

def solve_direct_rationale(
    llm: HFLLM,
    sample: QuestionSample,
    temperature: float = 0.0
) -> Dict[str, Any]:
    prompt = build_direct_rationale_prompt(sample)
    gen_result = llm.generate(prompt, temperature=temperature)

    try:
        parsed = parse_direct_rationale_output(
            llm=llm,
            raw_output=gen_result.text,
            option_texts=sample.options,
            input_tokens=gen_result.input_tokens,
            output_tokens=gen_result.output_tokens,
            total_tokens=gen_result.total_tokens,
            temperature=temperature
        )
    except Exception as e:
        usage = {
            "num_calls": 1,
            "input_tokens": gen_result.input_tokens,
            "output_tokens": gen_result.output_tokens,
            "total_tokens": gen_result.total_tokens
        }
        raise DirectRationaleExecutionError(
            message=f"DirectRationale parse failed: {e}",
            usage=usage,
            raw_output=gen_result.text
        ) from e

    return {
        "prediction": parsed.answer,
        "rationale": parsed.rationale,
        "raw_output": parsed.raw_output,
        "input_tokens": parsed.input_tokens,
        "output_tokens": parsed.output_tokens,
        "total_tokens": parsed.total_tokens,
        "num_calls": parsed.num_calls,
    }


# =========================================================
# Evaluation
# =========================================================

def evaluate_direct_rationale(
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
    total_calls = 0

    skipped_samples = []

    for idx, sample in enumerate(dataset):
        try:
            output = solve_direct_rationale(llm, sample, temperature=temperature)
            pred_label = output["prediction"]

            is_correct = None if sample.answer is None else (pred_label == sample.answer)

            row = {
                "subject": sample.subject,
                "question": sample.question,
                "options": sample.options,
                "prediction": pred_label,
                "gold": sample.answer,
                "correct": is_correct,
                "rationale": output["rationale"],
                "raw_output": output["raw_output"],
                "num_calls": output["num_calls"],
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

            total_input_tokens += output["input_tokens"]
            total_output_tokens += output["output_tokens"]
            total_tokens += output["total_tokens"]
            total_calls += output["num_calls"]

            if verbose:
                acc_so_far = correct / total if total > 0 else 0.0
                print("====================================================")
                print(output["rationale"])
                print("====================================================")
                print(f"[{idx + 1}/{len(dataset)}]")
                print("subject:", sample.subject)
                print("prediction:", pred_label)
                print("gold:", sample.answer)
                print("correct:", is_correct)
                print("num_calls:", output["num_calls"])
                print("input_tokens:", output["input_tokens"])
                print("output_tokens:", output["output_tokens"])
                print("total_tokens:", output["total_tokens"])
                print("current_acc:", round(acc_so_far, 4))
                print("====================================================")

        except DirectRationaleExecutionError as e:
            skipped += 1
            total += 1
            subject_stats[sample.subject]["total"] += 1

            total_input_tokens += e.usage["input_tokens"]
            total_output_tokens += e.usage["output_tokens"]
            total_tokens += e.usage["total_tokens"]
            total_calls += e.usage["num_calls"]

            skipped_samples.append({
                "index": idx,
                "subject": sample.subject,
                "question": sample.question,
                "options": sample.options,
                "gold": sample.answer,
                "error": str(e),
                "raw_output": e.raw_output,
                "num_calls": e.usage["num_calls"],
                "input_tokens": e.usage["input_tokens"],
                "output_tokens": e.usage["output_tokens"],
                "total_tokens": e.usage["total_tokens"],
            })

            print(f"[Error] sample index={idx}, reason={e}")

        except Exception as e:
            skipped += 1
            total += 1
            subject_stats[sample.subject]["total"] += 1

            skipped_samples.append({
                "index": idx,
                "subject": sample.subject,
                "question": sample.question,
                "options": sample.options,
                "gold": sample.answer,
                "error": str(e),
                "raw_output": None,
                "num_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            })

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
        "total_calls": total_calls,
        "avg_input_tokens": total_input_tokens / total if total > 0 else None,
        "avg_output_tokens": total_output_tokens / total if total > 0 else None,
        "avg_total_tokens": total_tokens / total if total > 0 else None,
        "avg_num_calls": total_calls / total if total > 0 else None,
        "subject_accuracy": subject_accuracy,
        "results": predictions,
        "skipped_samples": skipped_samples,
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
        num_samples_per_subject=50
    )

    result = evaluate_direct_rationale(
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
    print("Total input tokens:", result["total_input_tokens"])
    print("Total output tokens:", result["total_output_tokens"])
    print("Total tokens:", result["total_tokens"])
    print("Total calls:", result["total_calls"])
    print("Avg input tokens:", result["avg_input_tokens"])
    print("Avg output tokens:", result["avg_output_tokens"])
    print("Avg total tokens:", result["avg_total_tokens"])
    print("Avg num calls:", result["avg_num_calls"])

    save_results_json(result, "direct_rationale_final.json")