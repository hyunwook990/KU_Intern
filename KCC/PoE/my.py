# final decision에 rationale, confidence, 제거된 선택지 정보 주지 않음
# 재정렬 적용 A, B, D가 남으면 A, B, C로 선택지를 재정렬 후 LLM에 전달, 이후 다시 A, B, D로 복원
# min confidence 전부 제거
import json
import re
import random
import torch

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Literal
from collections import defaultdict

from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================================
# 설정
# =========================================================

MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"


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
class ModuleAOptionResult:
    rationale: str
    confidence: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    num_calls: int = 1


@dataclass
class ModuleAResult:
    rationales: List[str]
    confidences: List[float]
    num_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int


@dataclass
class FirstEliminationResult:
    elimination_mask: List[int]
    remaining_indices: List[int]


@dataclass
class FinalDecisionResult:
    final_answer_label: str
    final_explanation: str
    calibrated_confidence: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    num_calls: int = 1
    used_fallback: bool = False


@dataclass
class EliminatedOptionRecord:
    global_index: int
    label: str
    option_text: str
    rationale: str
    confidence: float
    eliminated_stage: str


# =========================================================
# Exception Classes
# =========================================================

class ModuleExecutionError(Exception):
    def __init__(self, module_name: str, usage: Dict, message: str):
        super().__init__(message)
        self.module_name = module_name
        self.usage = usage


class PipelineExecutionError(Exception):
    def __init__(self, message: str, usage: Dict, trace: Optional[List] = None):
        super().__init__(message)
        self.usage = usage
        self.trace = trace if trace is not None else []


# =========================================================
# Prompt Templates
# =========================================================

MODULE_A_PROMPT = """당신은 객관식 문제의 선택지를 평가하는 전문가입니다.

문제:
{question}

선택지:
{target_label}. {target_option}

작업:
1. 문제의 요구사항을 먼저 파악하세요.
2. 이 선택지가 문제의 요구사항을 충족하는지 단계적으로 검토하세요.
3. 검토 결과를 바탕으로 이 선택지가 정답일 가능성을 confidence 값으로 평가하세요.
4. 내부 추론 과정은 길게 쓰지 말고, 핵심 판단 근거만 1~2문장 rationale로 요약하세요.

반드시 아래 JSON 형식으로만 답하세요:
{{
  "rationale": "핵심 판단 근거 요약",
  "confidence": 0.0
}}

규칙:
- confidence는 이 선택지가 정답일 확률입니다.
- 0과 1 사이의 실수로 답하세요.
- rationale에는 최종 판단에 필요한 핵심 근거만 쓰세요.
- 단계별 사고 과정이나 장황한 추론은 출력하지 마세요.
- JSON 이외의 텍스트는 출력하지 마세요.
"""


FINAL_DECISION_PROMPT = """당신은 객관식 문제를 푸는 전문가입니다.

문제:
{question}

선택지:
{remaining_candidates_text}

작업:
1. 문제의 핵심 개념을 파악하세요.
2. 각 선택지를 비교하며 맞는지/틀린지 간단히 검토하세요.
3. 그 검토를 바탕으로 최종 정답 하나를 고르세요.

반드시 아래 JSON 형식으로만 답하세요:
{{
  "reasoning": "Step 1: ... Step 2: ... Step 3: ...",
  "answer": "A"
}}

규칙:
- reasoning에는 step-by-step 판단 과정을 쓰세요.
- reasoning은 비어 있으면 안 됩니다.
- answer에는 반드시 하나의 선택지 라벨만 넣으세요.
- answer에는 선택지 내용 전체나 일부를 쓰지 말고 라벨만 쓰세요.
- JSON 이외의 텍스트는 출력하지 마세요.
"""


REPAIR_MODULE_A_PROMPT = """아래 출력은 형식이 깨졌거나 JSON 파싱이 실패했습니다.
의미는 최대한 유지하고 반드시 JSON만 다시 출력하세요.

원래 출력:
{raw_output}

반드시 아래 형식으로만 출력하세요:
{{
  "rationale": "설명",
  "confidence": 0.0
}}

규칙:
- JSON 이외의 텍스트는 절대 출력하지 마세요.
- rationale은 비어 있으면 안 됩니다.
- confidence는 0과 1 사이의 실수여야 합니다.
"""


REPAIR_FINAL_DECISION_PROMPT = """아래 출력은 형식이 깨졌거나 JSON 파싱이 실패했습니다.
의미는 최대한 유지하고 반드시 JSON만 다시 출력하세요.

원래 출력:
{raw_output}

반드시 아래 형식으로만 출력하세요:
{{
  "reasoning": "Step 1: ... Step 2: ... Step 3: ...",
  "answer": "A"
}}

규칙:
- JSON 이외의 텍스트는 절대 출력하지 마세요.
- reasoning은 비어 있으면 안 됩니다.
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
        max_new_tokens: int = 300,
    ):
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

    while i < len(raw):
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
            if i + 1 >= len(raw):
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
    last_err = None

    for candidate in [raw, clean_json_like_string(raw)]:
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


def clamp_confidence(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def parse_confidence_value(value) -> float:
    if isinstance(value, (int, float)):
        return clamp_confidence(float(value))

    s = str(value).strip()
    has_percent = "%" in s
    s = s.replace("%", "").replace(",", "")

    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        raise ValueError(f"Cannot parse confidence: {value}")

    num = float(m.group(0))
    if has_percent or num > 1.0:
        num /= 100.0

    return clamp_confidence(num)


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
    remaining_labels: List[str],
    remaining_texts: List[str]
) -> str:
    try:
        return normalize_label(str(raw_answer), valid_labels=remaining_labels)
    except Exception:
        pass

    answer_text = str(raw_answer).strip()
    norm_answer = normalize_option_text_for_match(answer_text)

    for label, text in zip(remaining_labels, remaining_texts):
        if answer_text == str(text).strip():
            return label

    normalized_text_map = {
        label: normalize_option_text_for_match(text)
        for label, text in zip(remaining_labels, remaining_texts)
    }

    for label, norm_text in normalized_text_map.items():
        if norm_answer == norm_text:
            return label

    for label, text in zip(remaining_labels, remaining_texts):
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


def canonicalize_keys(data: dict) -> dict:
    key_aliases = {
        "rationale": "rationale",
        "reasoning": "rationale",
        "reason": "rationale",
        "explanation": "explanation",
        "confidence": "confidence",
        "score": "confidence",
        "probability": "confidence",
        "answer": "answer",
        "final_answer": "answer",
    }

    return {
        key_aliases.get(str(k).strip().lower(), str(k).strip().lower()): v
        for k, v in data.items()
    }


def repair_module_a_output(
    llm: HFLLM,
    raw_output: str,
    temperature: float = 0.0
) -> GenerationResult:
    prompt = REPAIR_MODULE_A_PROMPT.format(raw_output=raw_output)
    return llm.generate(prompt, temperature=temperature)


def repair_final_decision_output(
    llm: HFLLM,
    raw_output: str,
    temperature: float = 0.0
) -> GenerationResult:
    prompt = REPAIR_FINAL_DECISION_PROMPT.format(raw_output=raw_output)
    return llm.generate(prompt, temperature=temperature)


def parse_module_a_option_output_from_text(
    raw_output: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    num_calls: int = 1
) -> ModuleAOptionResult:
    data = canonicalize_keys(safe_json_loads(raw_output))

    rationale = str(data.get("rationale", "")).strip()
    confidence = parse_confidence_value(data["confidence"])

    if not rationale:
        raise ValueError(f"Empty rationale:\n{raw_output}")

    return ModuleAOptionResult(
        rationale=rationale,
        confidence=confidence,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        num_calls=num_calls
    )


def parse_module_a_option_output(
    llm: HFLLM,
    raw_output: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    temperature: float = 0.0
) -> ModuleAOptionResult:
    try:
        return parse_module_a_option_output_from_text(
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            num_calls=1
        )
    except Exception:
        repaired = repair_module_a_output(llm, raw_output, temperature=temperature)
        return parse_module_a_option_output_from_text(
            raw_output=repaired.text,
            input_tokens=input_tokens + repaired.input_tokens,
            output_tokens=output_tokens + repaired.output_tokens,
            total_tokens=total_tokens + repaired.total_tokens,
            num_calls=2
        )


def parse_final_decision_output_from_text(
    raw_output: str,
    remaining_labels: List[str],
    remaining_texts: List[str]
) -> Dict:
    data = canonicalize_keys(safe_json_loads(raw_output))

    raw_answer = data.get("answer")

    final_answer_label = resolve_answer_to_label(
        raw_answer=raw_answer,
        remaining_labels=remaining_labels,
        remaining_texts=remaining_texts
    )

    final_explanation = str(data.get("reasoning",data.get("explanation", data.get("rationale", "")))).strip()

    if not final_explanation:
        raise ValueError(f"Empty final explanation:\n{raw_output}")

    return {
        "final_answer_label": final_answer_label,
        "final_explanation": final_explanation,
    }


def parse_final_decision_output(
    llm: HFLLM,
    raw_output: str,
    remaining_labels: List[str],
    remaining_texts: List[str],
    temperature: float = 0.0
) -> Dict:
    try:
        parsed = parse_final_decision_output_from_text(
            raw_output=raw_output,
            remaining_labels=remaining_labels,
            remaining_texts=remaining_texts
        )
        parsed["repair_input_tokens"] = 0
        parsed["repair_output_tokens"] = 0
        parsed["repair_total_tokens"] = 0
        parsed["num_calls"] = 1
        return parsed
    except Exception:
        repaired = repair_final_decision_output(llm, raw_output, temperature=temperature)
        parsed = parse_final_decision_output_from_text(
            raw_output=repaired.text,
            remaining_labels=remaining_labels,
            remaining_texts=remaining_texts
        )
        parsed["repair_input_tokens"] = repaired.input_tokens
        parsed["repair_output_tokens"] = repaired.output_tokens
        parsed["repair_total_tokens"] = repaired.total_tokens
        parsed["num_calls"] = 2
        return parsed


def format_candidate_block(labels: List[str], options: List[str]) -> str:
    if not labels:
        return "없음"
    return "\n".join([f"- {label}. {opt}" for label, opt in zip(labels, options)])


# =========================================================
# 데이터셋 로드
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

    print(f"Loaded all subject\nTotal subject: {len(subjects)}")
    return all_samples


# =========================================================
# Module A
# =========================================================

class ModuleA:
    def __init__(self, llm: HFLLM):
        self.llm = llm

    def _build_prompt(
        self,
        question: str,
        options: List[str],
        prompt_template: str,
        target_idx: int
    ) -> str:
        return prompt_template.format(
            question=question,
            target_label=option_label(target_idx),
            target_option=options[target_idx]
        )

    def run(
        self,
        question: str,
        options: List[str],
        prompt_template: str,
        temperature: float = 0.0
    ) -> ModuleAResult:
        rationales = []
        confidences = []

        input_tokens_list = []
        output_tokens_list = []
        total_tokens_list = []

        total_num_calls = 0

        print("==============================Module_A==============================")

        for target_idx in range(len(options)):
            prompt = self._build_prompt(
                question=question,
                options=options,
                prompt_template=prompt_template,
                target_idx=target_idx
            )

            gen_result = self.llm.generate(prompt, temperature=temperature)

            try:
                parsed = parse_module_a_option_output(
                    llm=self.llm,
                    raw_output=gen_result.text,
                    input_tokens=gen_result.input_tokens,
                    output_tokens=gen_result.output_tokens,
                    total_tokens=gen_result.total_tokens,
                    temperature=temperature
                )
            except Exception as e:
                partial_usage = {
                    "num_calls": total_num_calls + 1,
                    "input_tokens": sum(input_tokens_list) + gen_result.input_tokens,
                    "output_tokens": sum(output_tokens_list) + gen_result.output_tokens,
                    "total_tokens": sum(total_tokens_list) + gen_result.total_tokens
                }

                raise ModuleExecutionError(
                    module_name="module_a",
                    usage=partial_usage,
                    message=f"ModuleA parse failed at option index={target_idx}: {e}"
                ) from e

            rationales.append(parsed.rationale)
            confidences.append(parsed.confidence)

            input_tokens_list.append(parsed.input_tokens)
            output_tokens_list.append(parsed.output_tokens)
            total_tokens_list.append(parsed.total_tokens)

            total_num_calls += parsed.num_calls

        return ModuleAResult(
            rationales=rationales,
            confidences=confidences,
            num_calls=total_num_calls,
            total_input_tokens=sum(input_tokens_list),
            total_output_tokens=sum(output_tokens_list),
            total_tokens=sum(total_tokens_list)
        )


# =========================================================
# First Elimination
# =========================================================

class FirstElimination:
    def __init__(
        self,
        mode: Literal["top1_ratio", "mean"] = "top1_ratio",
        top1_ratio: float = 0.8
    ):
        self.mode = mode
        self.top1_ratio = top1_ratio

    def run(
        self,
        options: List[str],
        confidences: List[float]
    ) -> FirstEliminationResult:
        print("==============================First_Elimination==============================")

        if len(options) != len(confidences):
            raise ValueError(
                f"options와 confidences 길이가 다릅니다: "
                f"len(options)={len(options)}, len(confidences)={len(confidences)}"
            )

        if not confidences:
            raise ValueError("confidence가 비어 있습니다.")

        top1_conf = max(confidences)

        if self.mode == "top1_ratio":
            threshold = top1_conf * self.top1_ratio
            if threshold > (top1_conf - 0.15):
                threshold = top1_conf - 0.15
        elif self.mode == "mean":
            threshold = sum(confidences) / len(confidences)
        elif self.mode == "min":
            threshold = min(confidences)
        else:
            raise ValueError(f"지원하지 않는 mode: {self.mode}")

        elimination_mask = []
        remaining_indices = []
            
        for idx, conf in enumerate(confidences):
            if conf <= threshold:
                elimination_mask.append(1)
            else:
                elimination_mask.append(0)
                remaining_indices.append(idx)

        if not remaining_indices:
            top_idx = max(range(len(confidences)), key=lambda i: confidences[i])
            elimination_mask = [1] * len(confidences)
            elimination_mask[top_idx] = 0
            remaining_indices = [top_idx]

        print(f"remaining_options: {[option_label(i) for i in remaining_indices]}")

        return FirstEliminationResult(
            elimination_mask=elimination_mask,
            remaining_indices=remaining_indices
        )


# =========================================================
# Final Decision
# =========================================================

class FinalDecision:
    def __init__(
        self,
        llm: HFLLM,
        calibration_fn: Optional[Callable[[float], float]] = None
    ):
        self.llm = llm
        self.calibration_fn = calibration_fn if calibration_fn is not None else (lambda x: x)

    def run(
        self,
        question: str,
        remaining_labels: List[str],
        remaining_texts: List[str],
        remaining_confidences: List[float],
        temperature: float = 0.0
    ) -> FinalDecisionResult:
        print("==============================Final_Decision==============================")

        remaining_candidates_text = format_candidate_block(
            labels=remaining_labels,
            options=remaining_texts
        )

        prompt = FINAL_DECISION_PROMPT.format(
            question=question,
            remaining_candidates_text=remaining_candidates_text
        )

        gen_result = self.llm.generate(prompt, temperature=temperature)

        try:
            parsed = parse_final_decision_output(
                llm=self.llm,
                raw_output=gen_result.text,
                remaining_labels=remaining_labels,
                remaining_texts=remaining_texts,
                temperature=temperature
            )

            final_answer_label = parsed["final_answer_label"]
            final_explanation = parsed["final_explanation"]

            chosen_idx = remaining_labels.index(final_answer_label)
            base_confidence = remaining_confidences[chosen_idx]

            calibrated_confidence = clamp_confidence(
                self.calibration_fn(base_confidence)
            )

            total_input_tokens = gen_result.input_tokens + parsed["repair_input_tokens"]
            total_output_tokens = gen_result.output_tokens + parsed["repair_output_tokens"]
            total_tokens = gen_result.total_tokens + parsed["repair_total_tokens"]

            return FinalDecisionResult(
                final_answer_label=final_answer_label,
                final_explanation=final_explanation,
                calibrated_confidence=calibrated_confidence,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_tokens,
                num_calls=parsed["num_calls"],
                used_fallback=False
            )

        except Exception:
            top_idx = max(range(len(remaining_confidences)), key=lambda i: remaining_confidences[i])
            fallback_label = remaining_labels[top_idx]
            fallback_confidence = remaining_confidences[top_idx]

            return FinalDecisionResult(
                final_answer_label=fallback_label,
                final_explanation=(
                    f"{final_explanation}"
                    f"FinalDecision 파싱에 실패하여 fallback을 적용했습니다. "
                    f"남아 있는 후보 중 confidence가 가장 높은 {fallback_label}를 최종 정답으로 선택했습니다."
                ),
                calibrated_confidence=clamp_confidence(self.calibration_fn(fallback_confidence)),
                input_tokens=gen_result.input_tokens,
                output_tokens=gen_result.output_tokens,
                total_tokens=gen_result.total_tokens,
                num_calls=1,
                used_fallback=True
            )


# =========================================================
# Pipeline
# =========================================================

class EliminationPipeline:
    def __init__(
        self,
        llm: HFLLM,
        module_a_prompt_template: str = MODULE_A_PROMPT,
        first_elimination_mode: Literal["top1_ratio", "mean"] = "top1_ratio",
        top1_ratio: float = 0.8,
        calibration_fn: Optional[Callable[[float], float]] = None,
        tie_round_decimals: int = 6
    ):
        self.llm = llm
        self.module_a_prompt_template = module_a_prompt_template
        self.tie_round_decimals = tie_round_decimals

        self.module_a = ModuleA(llm)
        self.first_elimination = FirstElimination(
            mode=first_elimination_mode,
            top1_ratio=top1_ratio
        )
        self.final_decision = FinalDecision(
            llm=llm,
            calibration_fn=calibration_fn
        )

    def _init_usage(self) -> Dict:
        return {
            "num_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0
        }

    def _add_usage(
        self,
        usage: Dict,
        num_calls: int,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int
    ) -> None:
        usage["num_calls"] += num_calls
        usage["total_input_tokens"] += input_tokens
        usage["total_output_tokens"] += output_tokens
        usage["total_tokens"] += total_tokens

    def _add_partial_usage_from_error(
        self,
        usage: Dict,
        err: ModuleExecutionError
    ) -> None:
        usage["num_calls"] += err.usage["num_calls"]
        usage["total_input_tokens"] += err.usage["input_tokens"]
        usage["total_output_tokens"] += err.usage["output_tokens"]
        usage["total_tokens"] += err.usage["total_tokens"]

    def _subset_by_indices(self, values: List, indices: List[int]) -> List:
        return [values[i] for i in indices]

    def _labels_from_global_indices(self, global_indices: List[int]) -> List[str]:
        return [option_label(i) for i in global_indices]

    def _restore_top2_with_ties_if_single_remaining(
        self,
        remaining_local_indices: List[int],
        confidences: List[float]
    ) -> List[int]:
        if len(remaining_local_indices) != 1:
            return remaining_local_indices

        if len(confidences) < 2:
            return remaining_local_indices

        rounded_confidences = [
            round(float(c), self.tie_round_decimals)
            for c in confidences
        ]

        sorted_unique_scores = sorted(set(rounded_confidences), reverse=True)

        if len(sorted_unique_scores) >= 2:
            cutoff_score = sorted_unique_scores[1]
        else:
            cutoff_score = sorted_unique_scores[0]

        restored_indices = [
            i for i, conf in enumerate(rounded_confidences)
            if conf >= cutoff_score
        ]

        print(
            "single remaining detected -> restored top-2 score group with ties:",
            [option_label(i) for i in restored_indices],
            "cutoff_score:",
            cutoff_score
        )

        return restored_indices

    def _collect_eliminated_records(
        self,
        current_options: List[str],
        current_global_indices: List[int],
        rationales: List[str],
        confidences: List[float],
        elimination_mask: List[int],
        stage_name: str
    ) -> List[EliminatedOptionRecord]:
        records = []

        for local_idx, eliminated in enumerate(elimination_mask):
            if eliminated != 1:
                continue

            global_idx = current_global_indices[local_idx]
            records.append(
                EliminatedOptionRecord(
                    global_index=global_idx,
                    label=option_label(global_idx),
                    option_text=current_options[local_idx],
                    rationale=rationales[local_idx],
                    confidence=confidences[local_idx],
                    eliminated_stage=stage_name
                )
            )

        return records

    def run(
        self,
        sample: QuestionSample,
        prompt_template: Optional[str] = None,
        temperature: float = 0.0
    ) -> Dict:
        actual_prompt_template = prompt_template or self.module_a_prompt_template

        current_options = list(sample.options)
        current_global_indices = list(range(len(sample.options)))

        trace = []
        sample_usage = self._init_usage()

        try:
            a_result = self.module_a.run(
                question=sample.question,
                options=current_options,
                prompt_template=actual_prompt_template,
                temperature=temperature
            )

            self._add_usage(
                usage=sample_usage,
                num_calls=a_result.num_calls,
                input_tokens=a_result.total_input_tokens,
                output_tokens=a_result.total_output_tokens,
                total_tokens=a_result.total_tokens
            )

            first_result = self.first_elimination.run(
                options=current_options,
                confidences=a_result.confidences
            )

            first_eliminated = self._collect_eliminated_records(
                current_options=current_options,
                current_global_indices=current_global_indices,
                rationales=a_result.rationales,
                confidences=a_result.confidences,
                elimination_mask=first_result.elimination_mask,
                stage_name="first_filter"
            )

            gold_removed = False
            if sample.answer is not None:
                eliminated_labels = [r.label for r in first_eliminated]
                gold_removed = sample.answer in eliminated_labels

            original_remaining_local_indices = list(first_result.remaining_indices)
            original_remaining_global_indices = self._subset_by_indices(
                current_global_indices,
                original_remaining_local_indices
            )
            original_remaining_labels = self._labels_from_global_indices(
                original_remaining_global_indices
            )

            single_remaining_restored = len(original_remaining_local_indices) == 1

            remaining_local_indices = self._restore_top2_with_ties_if_single_remaining(
                remaining_local_indices=original_remaining_local_indices,
                confidences=a_result.confidences
            )

            remaining_global_indices = self._subset_by_indices(
                current_global_indices,
                remaining_local_indices
            )
            remaining_labels = self._labels_from_global_indices(remaining_global_indices)
            remaining_options = self._subset_by_indices(current_options, remaining_local_indices)
            remaining_confidences = self._subset_by_indices(a_result.confidences, remaining_local_indices)
            
            original_remaining_labels_for_final = list(remaining_labels)

            reordered_remaining_labels = [
                option_label(i)
                for i in range(len(remaining_labels))
            ]

            needs_reordering = (
                original_remaining_labels_for_final != reordered_remaining_labels
            )

            if needs_reordering:
                labels_for_llm = reordered_remaining_labels

                final_to_original_label = dict(
                    zip(labels_for_llm, original_remaining_labels_for_final)
                )

                original_to_final_label = dict(
                    zip(original_remaining_labels_for_final, labels_for_llm)
                )
            else:
                labels_for_llm = original_remaining_labels_for_final

                final_to_original_label = {
                    label: label
                    for label in original_remaining_labels_for_final
                }

                original_to_final_label = {
                    label: label
                    for label in original_remaining_labels_for_final
                }
            if needs_reordering:
                print(
                    "remaining_index reordering:",
                    original_remaining_labels_for_final,
                    "->",
                    labels_for_llm,
                    )
                print(
                    "final_to_original_label:",
                    final_to_original_label
                    )
            gold_restored_after_single_remaining = False
            gold_only_remaining_before_restore = False

            if sample.answer is not None and single_remaining_restored:
                gold_was_removed_before_restore = sample.answer not in original_remaining_labels
                gold_exists_after_restore = sample.answer in remaining_labels

                gold_restored_after_single_remaining = (
                    gold_was_removed_before_restore
                    and gold_exists_after_restore
                )

                gold_only_remaining_before_restore = (
                    len(original_remaining_labels) == 1
                    and original_remaining_labels[0] == sample.answer
                )

            trace.append({
                "module_a": {
                    "rationales": a_result.rationales,
                    "confidences": a_result.confidences
                },
                "first_elimination": {
                    "elimination_mask": first_result.elimination_mask,
                    "original_remaining_labels": original_remaining_labels,
                    "final_decision_candidate_original_labels": original_remaining_labels_for_final,
                    "final_decision_candidate_labels_for_llm": labels_for_llm,
                    "single_remaining_restored": single_remaining_restored,
                    "label_reordering": {
                        "original_remaining_labels": original_remaining_labels_for_final,
                        "applied": needs_reordering,
                        "labels_for_llm": labels_for_llm,
                        "reordered_remaining_labels": reordered_remaining_labels,
                        "final_to_original_label": final_to_original_label,
                        "original_to_final_label": original_to_final_label,
                        "display": (
                            f"{original_remaining_labels_for_final} -> "
                            f"{labels_for_llm}"
                        )
                    },
                    "gold_restored_after_single_remaining": gold_restored_after_single_remaining,
                    "gold_only_remaining_before_restore": gold_only_remaining_before_restore,
                    "eliminated_records": [
                        {
                            "label": r.label,
                            "confidence": r.confidence
                        }
                        for r in first_eliminated
                    ]
                }
            })

            final_result = self.final_decision.run(
                question=sample.question,
                remaining_labels=labels_for_llm,
                remaining_texts=remaining_options,
                remaining_confidences=remaining_confidences,
                temperature=temperature
            )
            
            final_answer_llm_label = final_result.final_answer_label
            final_answer_original_label = final_to_original_label[final_answer_llm_label]

            self._add_usage(
                usage=sample_usage,
                num_calls=final_result.num_calls,
                input_tokens=final_result.input_tokens,
                output_tokens=final_result.output_tokens,
                total_tokens=final_result.total_tokens
            )

            restored_gold_selected = (
                gold_restored_after_single_remaining
                and final_answer_original_label == sample.answer
            )

            gold_only_remaining_then_lost = (
                gold_only_remaining_before_restore
                and final_answer_original_label != sample.answer
            )

            gold_only_remaining_and_still_selected = (
                gold_only_remaining_before_restore
                and final_answer_original_label == sample.answer
            )

            trace[-1]["final_decision"] = {
                "candidate_labels_for_llm": labels_for_llm,
                "candidate_original_labels": original_remaining_labels_for_final,
                "candidate_confidences": remaining_confidences,
                "final_answer_llm_label": final_answer_llm_label,
                "final_answer_original_label": final_answer_original_label,
                "final_explanation": final_result.final_explanation,
            }

            return {
                "trace": trace,
                "final": {
                    "answer_label": final_answer_original_label,
                    "answer_text": sample.options[label_to_index(final_answer_original_label)],
                    "final_explanation": final_result.final_explanation,
                    "confidence": final_result.calibrated_confidence,
                    "used_fallback": final_result.used_fallback
                },
                "usage": sample_usage,
                "gold_removed_in_first_elimination": gold_removed,
                "single_remaining_restored": single_remaining_restored,
                "gold_restored_after_single_remaining": gold_restored_after_single_remaining,
                "restored_gold_selected": restored_gold_selected,
                "gold_only_remaining_before_restore": gold_only_remaining_before_restore,
                "gold_only_remaining_then_lost": gold_only_remaining_then_lost,
                "gold_only_remaining_and_still_selected": gold_only_remaining_and_still_selected,
            }

        except ModuleExecutionError as e:
            self._add_partial_usage_from_error(sample_usage, e)

            raise PipelineExecutionError(
                message=str(e),
                usage=sample_usage,
                trace=trace
            ) from e

        except Exception as e:
            raise PipelineExecutionError(
                message=str(e),
                usage=sample_usage,
                trace=trace
            ) from e


# =========================================================
# Evaluation
# =========================================================

def evaluate_dataset(
    pipeline: EliminationPipeline,
    dataset: List[QuestionSample],
    prompt_template: Optional[str] = None,
    temperature: float = 0.0,
    verbose: bool = True
) -> Dict:
    results = []

    correct = 0
    total = 0
    skipped = 0
    fallback_count = 0
    first_elimination_gold_removed_count = 0

    single_remaining_restore_count = 0
    gold_restored_after_single_remaining_count = 0
    restored_gold_selected_count = 0
    gold_only_remaining_before_restore_count = 0
    gold_only_remaining_then_lost_count = 0
    gold_only_remaining_and_still_selected_count = 0

    subject_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    dataset_total_input_tokens = 0
    dataset_total_output_tokens = 0
    dataset_total_tokens = 0
    dataset_total_calls = 0

    for idx, sample in enumerate(dataset):
        try:
            output = pipeline.run(
                sample=sample,
                prompt_template=prompt_template,
                temperature=temperature
            )

            if output.get("gold_removed_in_first_elimination", False):
                first_elimination_gold_removed_count += 1

            if output.get("single_remaining_restored", False):
                single_remaining_restore_count += 1

            if output.get("gold_restored_after_single_remaining", False):
                gold_restored_after_single_remaining_count += 1

            if output.get("restored_gold_selected", False):
                restored_gold_selected_count += 1

            if output.get("gold_only_remaining_before_restore", False):
                gold_only_remaining_before_restore_count += 1

            if output.get("gold_only_remaining_then_lost", False):
                gold_only_remaining_then_lost_count += 1

            if output.get("gold_only_remaining_and_still_selected", False):
                gold_only_remaining_and_still_selected_count += 1

            pred_label = output["final"]["answer_label"]
            usage = output["usage"]
            used_fallback = output["final"]["used_fallback"]

            row = {
                "subject": sample.subject,
                "question": sample.question,
                "options": sample.options,
                "prediction": pred_label,
                "gold": sample.answer,
                "correct": None if sample.answer is None else pred_label == sample.answer,
                "final_confidence": output["final"]["confidence"],
                "final_explanation": output["final"]["final_explanation"],
                "gold_removed_in_first_elimination": output["gold_removed_in_first_elimination"],
                "single_remaining_restored": output["single_remaining_restored"],
                "gold_restored_after_single_remaining": output["gold_restored_after_single_remaining"],
                "restored_gold_selected": output["restored_gold_selected"],
                "gold_only_remaining_before_restore": output["gold_only_remaining_before_restore"],
                "gold_only_remaining_then_lost": output["gold_only_remaining_then_lost"],
                "gold_only_remaining_and_still_selected": output["gold_only_remaining_and_still_selected"],
                "trace": output["trace"]
            }

            results.append(row)

            if used_fallback:
                fallback_count += 1

            if sample.answer is not None:
                total += 1
                correct += int(pred_label == sample.answer)
                subject_stats[sample.subject]["total"] += 1
                subject_stats[sample.subject]["correct"] += int(row["correct"])

            dataset_total_input_tokens += usage["total_input_tokens"]
            dataset_total_output_tokens += usage["total_output_tokens"]
            dataset_total_tokens += usage["total_tokens"]
            dataset_total_calls += usage["num_calls"]

            if verbose:
                acc_so_far = correct / total if total > 0 else 0.0

                print("#############################################################")
                print(output["final"]["final_explanation"])
                print("#############################################################")
                print("subject:", sample.subject)
                print("question:", sample.question)
                print("prediction:", pred_label)
                print("gold:", sample.answer)
                print("num_calls:", usage["num_calls"])
                print("total_input_tokens:", usage["total_input_tokens"])
                print("total_output_tokens:", usage["total_output_tokens"])
                print("total_tokens:", usage["total_tokens"])
                print("#############################################################")
                print(
                    f"[{idx + 1}/{len(dataset)}] "
                    f"current accuracy = {acc_so_far:.4f}, "
                    f"skipped = {skipped}, "
                    f"fallback = {fallback_count}"
                )

        except PipelineExecutionError as e:
            skipped += 1
            total += 1
            subject_stats[sample.subject]["total"] += 1

            dataset_total_input_tokens += e.usage["total_input_tokens"]
            dataset_total_output_tokens += e.usage["total_output_tokens"]
            dataset_total_tokens += e.usage["total_tokens"]
            dataset_total_calls += e.usage["num_calls"]

            print(f"[Error] sample index={idx}, reason={e}")

        except Exception as e:
            skipped += 1
            total += 1
            subject_stats[sample.subject]["total"] += 1

            print(f"[Error] sample index={idx}, reason={e}")

    subject_accuracy = {}

    print("\n===== Subject-wise Accuracy =====")

    for subject, stats in subject_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else None

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

    avg_input_tokens = dataset_total_input_tokens / total if total > 0 else None
    avg_output_tokens = dataset_total_output_tokens / total if total > 0 else None
    avg_total_tokens = dataset_total_tokens / total if total > 0 else None
    avg_num_calls = dataset_total_calls / total if total > 0 else None

    return {
        "accuracy": accuracy,
        "subject_accuracy": subject_accuracy,
        "num_evaluated": total,
        "num_correct": correct,
        "num_skipped": skipped,
        "num_fallback": fallback_count,
        "first_elimination_gold_removed_count": first_elimination_gold_removed_count,

        "single_remaining_restore_count": single_remaining_restore_count,
        "gold_restored_after_single_remaining_count": gold_restored_after_single_remaining_count,
        "restored_gold_selected_count": restored_gold_selected_count,
        "gold_only_remaining_before_restore_count": gold_only_remaining_before_restore_count,
        "gold_only_remaining_then_lost_count": gold_only_remaining_then_lost_count,
        "gold_only_remaining_and_still_selected_count": gold_only_remaining_and_still_selected_count,

        "dataset_total_input_tokens": dataset_total_input_tokens,
        "dataset_total_output_tokens": dataset_total_output_tokens,
        "dataset_total_tokens": dataset_total_tokens,
        "dataset_total_calls": dataset_total_calls,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "avg_total_tokens": avg_total_tokens,
        "avg_num_calls": avg_num_calls,
        "results": results
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
        max_new_tokens=300
    )

    pipeline = EliminationPipeline(
        llm=llm,
        module_a_prompt_template=MODULE_A_PROMPT,
        first_elimination_mode="min",
        top1_ratio=0.8,
        calibration_fn=lambda x: x,
        tie_round_decimals=6
    )

    dataset = load_all_subjects_random(
        dataset_name="HAERAE-HUB/KMMLU",
        split="test",
        num_samples_per_subject=50
    )

    print(f"Total samples: {len(dataset)}")

    result = evaluate_dataset(
        pipeline=pipeline,
        dataset=dataset,
        prompt_template=MODULE_A_PROMPT,
        temperature=0.0,
        verbose=True
    )

    print("\n===== Final Result =====")
    print("Accuracy:", result["accuracy"])
    print("Num evaluated:", result["num_evaluated"])
    print("Num correct:", result["num_correct"])
    print("First elimination removed gold count:", result["first_elimination_gold_removed_count"])

    print("Single remaining restore count:", result["single_remaining_restore_count"])
    print(
        "Gold restored:",
        f'{result["gold_restored_after_single_remaining_count"]} / {result["single_remaining_restore_count"]}'
    )
    print(
        "Restored gold selected:",
        f'{result["restored_gold_selected_count"]} / {result["gold_restored_after_single_remaining_count"]}'
    )
    print(
        "Gold only remaining then lost:",
        f'{result["gold_only_remaining_then_lost_count"]} / {result["gold_only_remaining_before_restore_count"]}'
    )
    print(
        "Gold only remaining and still selected:",
        f'{result["gold_only_remaining_and_still_selected_count"]} / {result["gold_only_remaining_before_restore_count"]}'
    )

    print("Num skipped:", result["num_skipped"])
    print("Num fallback:", result["num_fallback"])
    print("Dataset total input tokens:", result["dataset_total_input_tokens"])
    print("Dataset total output tokens:", result["dataset_total_output_tokens"])
    print("Dataset total tokens:", result["dataset_total_tokens"])
    print("Dataset total calls:", result["dataset_total_calls"])
    print("Avg input tokens:", result["avg_input_tokens"])
    print("Avg output tokens:", result["avg_output_tokens"])
    print("Avg total tokens:", result["avg_total_tokens"])
    print("Avg num calls:", result["avg_num_calls"])

    save_results_json(result, "my_refine_rearrange_eliminate_min.json")