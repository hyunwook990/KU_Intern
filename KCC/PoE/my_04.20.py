# 마지막에 reasoning을 다시 하지않고 첫 rationale 사용
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Literal
import random
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
class ModuleAOptionResult:
    rationale: str
    confidence: float
    raw_output: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class ModuleAResult:
    rationales: List[str]
    confidences: List[float]
    raw_outputs: List[str]
    input_tokens_list: List[int]
    output_tokens_list: List[int]
    total_tokens_list: List[int]
    num_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int


@dataclass
class FirstEliminationResult:
    elimination_mask: List[int]         # 0: keep, 1: eliminate
    remaining_indices: List[int]
    criterion_name: str
    criterion_value: float


@dataclass
class FinalDecisionResult:
    final_answer_label: str
    final_rationale: str
    calibrated_confidence: float
    raw_output: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    num_calls: int = 1


@dataclass
class EliminatedOptionRecord:
    global_index: int
    label: str
    option_text: str
    rationale: str
    confidence: float
    eliminated_stage: str   # "first_filter", "additional_filter", "compare_two_loser"


# =========================================================
# Exception Classes
# =========================================================

class ModuleExecutionError(Exception):
    def __init__(self, module_name: str, usage: Dict, message: str, raw_output: Optional[str] = None):
        super().__init__(message)
        self.module_name = module_name
        self.usage = usage
        self.raw_output = raw_output


class PipelineExecutionError(Exception):
    def __init__(self, message: str, usage: Dict, trace: Optional[List] = None):
        super().__init__(message)
        self.usage = usage
        self.trace = trace if trace is not None else []


# =========================================================
# Prompt Templates
# =========================================================

MODULE_A_PROMPT = """당신은 객관식 문제의 하나의 선택지만 평가하는 전문가입니다.

문제:
{question}

선택지:
{target_label}. {target_option}

작업:
1. 이 선택지가 정답일 가능성을 평가하세요.
2. 1~2문장으로 rationale을 작성하세요.
3. 이 선택지가 틀릴 수 있는 이유 또는 한계를 반드시 포함하세요.

반드시 아래 JSON 형식으로만 답하세요:
{{
  "rationale": "설명",
  "confidence": 0.0
}}

규칙:
- confidence는 이 선택지가 정답일 확률입니다.
- 0과 1 사이의 실수로 답하세요.
- JSON 이외의 텍스트는 출력하지 마세요.
"""


FINAL_DECISION_PROMPT = """당신은 객관식 문제의 최종 답안을 결정하는 전문가입니다.

문제:
{question}

현재 살아남은 후보 선택지들:
{remaining_candidates_text}

후보 선택지별 rationale:
{remaining_rationales_text}

이미 제거된 선택지들:
{eliminated_candidates_text}

제거된 선택지들의 rationale:
{eliminated_rationales_text}

작업:
1. 살아남은 후보들 중 최종 정답 하나를 고르세요.
2. 제거된 선택지들이 왜 탈락했는지도 참고해서 최종 판단하세요.
3. 특히 후보가 2개라면, 두 후보만 보지 말고 제거된 선택지들의 reasoning도 함께 보고 상대적으로 더 타당한 답을 고르세요.

반드시 아래 JSON 형식으로만 답하세요:
{{
  "answer": "라벨",
  "confidence": 0.0
}}

규칙:
- answer는 반드시 현재 살아남은 후보 라벨 중 하나여야 합니다.
- answer는 반드시 라벨만 출력해야 합니다.
- confidence는 최종 답안에 대한 확신도입니다.
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
        max_new_tokens: int = 300,
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


def clamp_confidence(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


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


def parse_module_a_option_output(
    raw_output: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int
) -> ModuleAOptionResult:
    data = safe_json_loads(raw_output)

    rationale = str(data["rationale"]).strip()
    confidence = clamp_confidence(float(data["confidence"]))

    return ModuleAOptionResult(
        rationale=rationale,
        confidence=confidence,
        raw_output=raw_output,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens
    )


def format_reason_block(
    labels: List[str],
    options: List[str],
    rationales: List[str],
    confidences: Optional[List[float]] = None
) -> str:
    if len(labels) == 0:
        return "없음"

    lines = []
    for i in range(len(labels)):
        if confidences is None:
            lines.append(f"- {labels[i]}. {options[i]}: {rationales[i]}")
        else:
            lines.append(
                f"- {labels[i]}. {options[i]}: {rationales[i]} (confidence={confidences[i]:.4f})"
            )
    return "\n".join(lines)


def format_candidate_block(labels: List[str], options: List[str]) -> str:
    if len(labels) == 0:
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

    num_samples = min(num_samples, len(ds))
    selected_indices = indices[:num_samples]

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
# Module A: 각 보기별 rationale + confidence
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
        labels = [option_label(i) for i in range(len(options))]
        return prompt_template.format(
            question=question,
            target_label=labels[target_idx],
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
        raw_outputs = []

        input_tokens_list = []
        output_tokens_list = []
        total_tokens_list = []

        print("==============================Module_A==============================")
        for target_idx in range(len(options)):
            prompt = self._build_prompt(
                question=question,
                options=options,
                prompt_template=prompt_template,
                target_idx=target_idx
            )

            gen_result = self.llm.generate(prompt, temperature=temperature)

            input_tokens_list.append(gen_result.input_tokens)
            output_tokens_list.append(gen_result.output_tokens)
            total_tokens_list.append(gen_result.total_tokens)
            raw_outputs.append(gen_result.text)

            try:
                parsed = parse_module_a_option_output(
                    gen_result.text,
                    gen_result.input_tokens,
                    gen_result.output_tokens,
                    gen_result.total_tokens
                )
            except Exception as e:
                partial_usage = {
                    "num_calls": len(total_tokens_list),
                    "input_tokens": sum(input_tokens_list),
                    "output_tokens": sum(output_tokens_list),
                    "total_tokens": sum(total_tokens_list)
                }
                raise ModuleExecutionError(
                    module_name="module_a",
                    usage=partial_usage,
                    message=f"ModuleA parse failed at option index={target_idx}: {e}",
                    raw_output=gen_result.text
                ) from e

            rationales.append(parsed.rationale)
            confidences.append(parsed.confidence)

        return ModuleAResult(
            rationales=rationales,
            confidences=confidences,
            raw_outputs=raw_outputs,
            input_tokens_list=input_tokens_list,
            output_tokens_list=output_tokens_list,
            total_tokens_list=total_tokens_list,
            num_calls=len(options),
            total_input_tokens=sum(input_tokens_list),
            total_output_tokens=sum(output_tokens_list),
            total_tokens=sum(total_tokens_list)
        )


# =========================================================
# First Elimination: top1 ratio or average
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

        if len(confidences) == 0:
            raise ValueError("confidence가 비어 있습니다.")

        top1_conf = max(confidences)
        mean_conf = sum(confidences) / len(confidences)

        if self.mode == "top1_ratio":
            threshold = top1_conf * self.top1_ratio
            criterion_name = f"top1_ratio({self.top1_ratio})"
        elif self.mode == "mean":
            threshold = mean_conf
            criterion_name = "mean"
        else:
            raise ValueError(f"지원하지 않는 mode: {self.mode}")

        elimination_mask = []
        remaining_indices = []

        for idx, conf in enumerate(confidences):
            if conf < threshold:
                elimination_mask.append(1)
            else:
                elimination_mask.append(0)
                remaining_indices.append(idx)

        # 안전장치: 다 제거되면 top1 하나는 살림
        if len(remaining_indices) == 0:
            top_idx = max(range(len(confidences)), key=lambda i: confidences[i])
            elimination_mask = [1] * len(confidences)
            elimination_mask[top_idx] = 0
            remaining_indices = [top_idx]

        print(f"remaining_options: {[chr(ord('A')+i) for i in remaining_indices]}")

        return FirstEliminationResult(
            elimination_mask=elimination_mask,
            remaining_indices=remaining_indices,
            criterion_name=criterion_name,
            criterion_value=threshold
        )


# =========================================================
# Final Decision
# explanation 새로 생성하지 않고,
# 선택된 보기의 Module A rationale을 그대로 최종 rationale로 사용
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
        remaining_rationales: List[str],
        remaining_confidences: List[float],
        eliminated_records: List[EliminatedOptionRecord],
        temperature: float = 0.0
    ) -> FinalDecisionResult:
        print("==============================Final_Decision==============================")

        remaining_candidates_text = format_candidate_block(
            labels=remaining_labels,
            options=remaining_texts
        )
        remaining_rationales_text = format_reason_block(
            labels=remaining_labels,
            options=remaining_texts,
            rationales=remaining_rationales,
            confidences=remaining_confidences
        )

        eliminated_labels = [r.label for r in eliminated_records]
        eliminated_options = [r.option_text for r in eliminated_records]
        eliminated_rationales = [r.rationale for r in eliminated_records]
        eliminated_confidences = [r.confidence for r in eliminated_records]

        eliminated_candidates_text = format_candidate_block(
            labels=eliminated_labels,
            options=eliminated_options
        )
        eliminated_rationales_text = format_reason_block(
            labels=eliminated_labels,
            options=eliminated_options,
            rationales=eliminated_rationales,
            confidences=eliminated_confidences
        )

        prompt = FINAL_DECISION_PROMPT.format(
            question=question,
            remaining_candidates_text=remaining_candidates_text,
            remaining_rationales_text=remaining_rationales_text,
            eliminated_candidates_text=eliminated_candidates_text,
            eliminated_rationales_text=eliminated_rationales_text
        )

        gen_result = self.llm.generate(prompt, temperature=temperature)

        try:
            data = safe_json_loads(gen_result.text)
            final_answer_label = str(data["answer"]).strip().upper()
            model_confidence = clamp_confidence(float(data["confidence"]))

            if final_answer_label not in remaining_labels:
                raise ValueError(
                    f"Invalid final answer: {final_answer_label}, valid={remaining_labels}"
                )

            chosen_idx = remaining_labels.index(final_answer_label)

            # 핵심 변경점:
            # 새 explanation을 생성하지 않고,
            # 선택된 보기의 첫 rationale(Module A 결과)을 그대로 사용
            final_rationale = remaining_rationales[chosen_idx]

            base_confidence = remaining_confidences[chosen_idx]
            calibrated_confidence = clamp_confidence(
                self.calibration_fn(max(base_confidence, model_confidence))
            )

        except Exception as e:
            raise ModuleExecutionError(
                module_name="final_decision",
                usage={
                    "num_calls": 1,
                    "input_tokens": gen_result.input_tokens,
                    "output_tokens": gen_result.output_tokens,
                    "total_tokens": gen_result.total_tokens
                },
                message=f"FinalDecision parse failed: {e}",
                raw_output=gen_result.text
            ) from e

        return FinalDecisionResult(
            final_answer_label=final_answer_label,
            final_rationale=final_rationale,
            calibrated_confidence=calibrated_confidence,
            raw_output=gen_result.text,
            input_tokens=gen_result.input_tokens,
            output_tokens=gen_result.output_tokens,
            total_tokens=gen_result.total_tokens
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
        calibration_fn: Optional[Callable[[float], float]] = None
    ):
        self.llm = llm
        self.module_a_prompt_template = module_a_prompt_template

        self.module_a = ModuleA(llm)
        self.first_elimination = FirstElimination(
            mode=first_elimination_mode,
            top1_ratio=top1_ratio
        )
        self.final_decision = FinalDecision(llm=llm, calibration_fn=calibration_fn)

    def _subset_by_indices(self, values: List, indices: List[int]) -> List:
        return [values[i] for i in indices]

    def _labels_from_global_indices(self, global_indices: List[int]) -> List[str]:
        return [option_label(i) for i in global_indices]

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

            gidx = current_global_indices[local_idx]
            records.append(
                EliminatedOptionRecord(
                    global_index=gidx,
                    label=option_label(gidx),
                    option_text=current_options[local_idx],
                    rationale=rationales[local_idx],
                    confidence=confidences[local_idx],
                    eliminated_stage=stage_name
                )
            )
        return records

    def _init_usage(self) -> Dict:
        return {
            "num_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "by_module": {
                "module_a": {
                    "num_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                },
                "final_decision": {
                    "num_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                }
            }
        }

    def _add_module_usage(
        self,
        usage: Dict,
        module_name: str,
        num_calls: int,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int
    ) -> None:
        usage["num_calls"] += num_calls
        usage["total_input_tokens"] += input_tokens
        usage["total_output_tokens"] += output_tokens
        usage["total_tokens"] += total_tokens

        usage["by_module"][module_name]["num_calls"] += num_calls
        usage["by_module"][module_name]["input_tokens"] += input_tokens
        usage["by_module"][module_name]["output_tokens"] += output_tokens
        usage["by_module"][module_name]["total_tokens"] += total_tokens

    def _add_partial_module_usage_from_error(
        self,
        usage: Dict,
        err: ModuleExecutionError
    ) -> None:
        module_name = err.module_name
        if module_name not in usage["by_module"]:
            raise ValueError(f"Unknown module_name in ModuleExecutionError: {module_name}")

        usage["num_calls"] += err.usage["num_calls"]
        usage["total_input_tokens"] += err.usage["input_tokens"]
        usage["total_output_tokens"] += err.usage["output_tokens"]
        usage["total_tokens"] += err.usage["total_tokens"]

        usage["by_module"][module_name]["num_calls"] += err.usage["num_calls"]
        usage["by_module"][module_name]["input_tokens"] += err.usage["input_tokens"]
        usage["by_module"][module_name]["output_tokens"] += err.usage["output_tokens"]
        usage["by_module"][module_name]["total_tokens"] += err.usage["total_tokens"]

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
        eliminated_records: List[EliminatedOptionRecord] = []
        sample_usage = self._init_usage()

        try:
            # -------------------------------------------------
            # 1) 각 보기마다 rationale + confidence
            # -------------------------------------------------
            a_result = self.module_a.run(
                question=sample.question,
                options=current_options,
                prompt_template=actual_prompt_template,
                temperature=temperature
            )

            self._add_module_usage(
                usage=sample_usage,
                module_name="module_a",
                num_calls=a_result.num_calls,
                input_tokens=a_result.total_input_tokens,
                output_tokens=a_result.total_output_tokens,
                total_tokens=a_result.total_tokens
            )

            # -------------------------------------------------
            # 2) 1차 제거: top1 ratio or mean
            # -------------------------------------------------
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
            eliminated_records.extend(first_eliminated)

            remaining_local_indices = first_result.remaining_indices
            remaining_global_indices = self._subset_by_indices(current_global_indices, remaining_local_indices)
            remaining_labels = self._labels_from_global_indices(remaining_global_indices)
            remaining_options = self._subset_by_indices(current_options, remaining_local_indices)
            remaining_rationales = self._subset_by_indices(a_result.rationales, remaining_local_indices)
            remaining_confidences = self._subset_by_indices(a_result.confidences, remaining_local_indices)

            trace.append({
                "module_a": {
                    "rationales": a_result.rationales,
                    "confidences": a_result.confidences,
                    "raw_outputs": a_result.raw_outputs
                },
                "first_elimination": {
                    "criterion_name": first_result.criterion_name,
                    "criterion_value": first_result.criterion_value,
                    "elimination_mask": first_result.elimination_mask,
                    "remaining_labels": remaining_labels,
                    "remaining_options": remaining_options,
                    "remaining_rationales": remaining_rationales,
                    "remaining_confidences": remaining_confidences,
                    "eliminated_records": [
                        {
                            "label": r.label,
                            "option_text": r.option_text,
                            "rationale": r.rationale,
                            "confidence": r.confidence,
                            "eliminated_stage": r.eliminated_stage
                        }
                        for r in first_eliminated
                    ]
                }
            })

            # -------------------------------------------------
            # 3) 최종 선택
            # explanation은 새로 생성하지 않고
            # 선택된 보기의 Module A rationale을 그대로 사용
            # -------------------------------------------------
            final_result = self.final_decision.run(
                question=sample.question,
                remaining_labels=remaining_labels,
                remaining_texts=remaining_options,
                remaining_rationales=remaining_rationales,
                remaining_confidences=remaining_confidences,
                eliminated_records=eliminated_records,
                temperature=temperature
            )

            self._add_module_usage(
                usage=sample_usage,
                module_name="final_decision",
                num_calls=final_result.num_calls,
                input_tokens=final_result.input_tokens,
                output_tokens=final_result.output_tokens,
                total_tokens=final_result.total_tokens
            )

            trace[-1]["final_decision"] = {
                "candidate_labels": remaining_labels,
                "candidate_options": remaining_options,
                "candidate_rationales": remaining_rationales,
                "candidate_confidences": remaining_confidences,
                "eliminated_records": [
                    {
                        "label": r.label,
                        "option_text": r.option_text,
                        "rationale": r.rationale,
                        "confidence": r.confidence,
                        "eliminated_stage": r.eliminated_stage
                    }
                    for r in eliminated_records
                ],
                "final_answer_label": final_result.final_answer_label,
                "final_rationale": final_result.final_rationale,
                "model_raw_output": final_result.raw_output
            }

            return {
                "trace": trace,
                "final": {
                    "answer_label": final_result.final_answer_label,
                    "answer_text": sample.options[label_to_index(final_result.final_answer_label)],
                    "final_explanation": final_result.final_rationale,
                    "confidence": final_result.calibrated_confidence
                },
                "usage": sample_usage
            }

        except ModuleExecutionError as e:
            self._add_partial_module_usage_from_error(sample_usage, e)
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

def init_dataset_module_usage() -> Dict:
    return {
        "module_a": {
            "num_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        },
        "final_decision": {
            "num_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
    }


def update_dataset_module_usage(dataset_module_usage: Dict, sample_module_usage: Dict) -> None:
    for module_name in ["module_a", "final_decision"]:
        dataset_module_usage[module_name]["num_calls"] += sample_module_usage[module_name]["num_calls"]
        dataset_module_usage[module_name]["input_tokens"] += sample_module_usage[module_name]["input_tokens"]
        dataset_module_usage[module_name]["output_tokens"] += sample_module_usage[module_name]["output_tokens"]
        dataset_module_usage[module_name]["total_tokens"] += sample_module_usage[module_name]["total_tokens"]


def compute_avg_module_usage(dataset_module_usage: Dict, denominator: int) -> Dict:
    if denominator == 0:
        return {
            "module_a": {
                "avg_num_calls": None,
                "avg_input_tokens": None,
                "avg_output_tokens": None,
                "avg_total_tokens": None
            },
            "final_decision": {
                "avg_num_calls": None,
                "avg_input_tokens": None,
                "avg_output_tokens": None,
                "avg_total_tokens": None
            }
        }

    avg_usage = {}
    for module_name in ["module_a", "final_decision"]:
        avg_usage[module_name] = {
            "avg_num_calls": dataset_module_usage[module_name]["num_calls"] / denominator,
            "avg_input_tokens": dataset_module_usage[module_name]["input_tokens"] / denominator,
            "avg_output_tokens": dataset_module_usage[module_name]["output_tokens"] / denominator,
            "avg_total_tokens": dataset_module_usage[module_name]["total_tokens"] / denominator
        }
    return avg_usage


def evaluate_dataset(
    pipeline: EliminationPipeline,
    dataset: List[QuestionSample],
    prompt_template: Optional[str] = None,
    temperature: float = 0.0,
    verbose: bool = True
) -> Dict:
    predictions = []
    correct = 0
    total = 0
    skipped = 0

    skipped_samples = []
    subject_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    dataset_total_input_tokens = 0
    dataset_total_output_tokens = 0
    dataset_total_tokens = 0
    dataset_total_calls = 0
    dataset_module_usage = init_dataset_module_usage()

    for idx, sample in enumerate(dataset):
        try:
            output = pipeline.run(
                sample=sample,
                prompt_template=prompt_template,
                temperature=temperature
            )

            pred_label = output["final"]["answer_label"]
            usage = output["usage"]

            row = {
                "subject": sample.subject,
                "question": sample.question,
                "options": sample.options,
                "prediction": pred_label,
                "gold": sample.answer,
                "correct": None if sample.answer is None else pred_label == sample.answer,
                "final_confidence": output["final"]["confidence"],
                "final_explanation": output["final"]["final_explanation"],
                "num_calls": usage["num_calls"],
                "total_input_tokens": usage["total_input_tokens"],
                "total_output_tokens": usage["total_output_tokens"],
                "total_tokens": usage["total_tokens"],
                "module_usage": usage["by_module"],
                "trace": output["trace"]
            }
            predictions.append(row)

            if sample.answer is not None:
                total += 1
                correct += int(pred_label == sample.answer)
                subject_stats[sample.subject]["total"] += 1
                subject_stats[sample.subject]["correct"] += int(row["correct"])

            dataset_total_input_tokens += usage["total_input_tokens"]
            dataset_total_output_tokens += usage["total_output_tokens"]
            dataset_total_tokens += usage["total_tokens"]
            dataset_total_calls += usage["num_calls"]

            update_dataset_module_usage(dataset_module_usage, usage["by_module"])

            if verbose:
                acc_so_far = correct / total if total > 0 else 0.0
                print("#############################################################")
                print(output["final"]["final_explanation"])
                print("#############################################################")
                print("subject", sample.subject)
                print("question:", sample.question)
                print("prediction:", pred_label)
                print("gold:", sample.answer)
                print("num_calls:", usage["num_calls"])
                print("total_input_tokens:", usage["total_input_tokens"])
                print("total_output_tokens:", usage["total_output_tokens"])
                print("total_tokens:", usage["total_tokens"])
                print("module_a_total_tokens:", usage["by_module"]["module_a"]["total_tokens"])
                print("final_decision_total_tokens:", usage["by_module"]["final_decision"]["total_tokens"])
                print("#############################################################")
                print(f"[{idx + 1}/{len(dataset)}] current accuracy = {acc_so_far:.4f}, skipped = {skipped}")

        except PipelineExecutionError as e:
            skipped += 1
            total += 1
            subject_stats[sample.subject]["total"] += 1

            dataset_total_input_tokens += e.usage["total_input_tokens"]
            dataset_total_output_tokens += e.usage["total_output_tokens"]
            dataset_total_tokens += e.usage["total_tokens"]
            dataset_total_calls += e.usage["num_calls"]

            update_dataset_module_usage(dataset_module_usage, e.usage["by_module"])

            skipped_samples.append({
                "index": idx,
                "subject": sample.subject,
                "question": sample.question,
                "options": sample.options,
                "gold": sample.answer,
                "error": str(e),
                "num_calls": e.usage["num_calls"],
                "total_input_tokens": e.usage["total_input_tokens"],
                "total_output_tokens": e.usage["total_output_tokens"],
                "total_tokens": e.usage["total_tokens"],
                "module_usage": e.usage["by_module"],
                "trace": e.trace
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
                "num_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "module_usage": None,
                "trace": []
            })

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

    acc = (correct / total) if total > 0 else None
    num_processed = len(predictions)
    num_attempted = total

    avg_input_tokens = dataset_total_input_tokens / num_attempted if num_attempted > 0 else None
    avg_output_tokens = dataset_total_output_tokens / num_attempted if num_attempted > 0 else None
    avg_total_tokens = dataset_total_tokens / num_attempted if num_attempted > 0 else None
    avg_num_calls = dataset_total_calls / num_attempted if num_attempted > 0 else None
    avg_module_usage = compute_avg_module_usage(dataset_module_usage, num_attempted)

    return {
        "accuracy": acc,
        "subject_accuracy": subject_accuracy,
        "num_evaluated": total,
        "num_correct": correct,
        "num_skipped": skipped,
        "num_processed": num_processed,
        "num_attempted": num_attempted,
        "dataset_total_input_tokens": dataset_total_input_tokens,
        "dataset_total_output_tokens": dataset_total_output_tokens,
        "dataset_total_tokens": dataset_total_tokens,
        "dataset_total_calls": dataset_total_calls,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "avg_total_tokens": avg_total_tokens,
        "avg_num_calls": avg_num_calls,
        "dataset_module_usage": dataset_module_usage,
        "avg_module_usage": avg_module_usage,
        "results": predictions,
        "skipped_samples": skipped_samples
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
        first_elimination_mode="top1_ratio",   # "top1_ratio" or "mean"
        top1_ratio=0.8,
        calibration_fn=lambda x: x
    )

    dataset = load_all_subjects_random(
        dataset_name="HAERAE-HUB/KMMLU",
        split="test",
        num_samples_per_subject=10
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
    print("Num skipped:", result["num_skipped"])
    print("Num processed:", result["num_processed"])
    print("Num attempted:", result["num_attempted"])
    print("Dataset total input tokens:", result["dataset_total_input_tokens"])
    print("Dataset total output tokens:", result["dataset_total_output_tokens"])
    print("Dataset total tokens:", result["dataset_total_tokens"])
    print("Dataset total calls:", result["dataset_total_calls"])
    print("Avg input tokens:", result["avg_input_tokens"])
    print("Avg output tokens:", result["avg_output_tokens"])
    print("Avg total tokens:", result["avg_total_tokens"])
    print("Avg num calls:", result["avg_num_calls"])

    print("\n===== Dataset Module Usage =====")
    print("Module A:", result["dataset_module_usage"]["module_a"])
    print("Final Decision:", result["dataset_module_usage"]["final_decision"])

    print("\n===== Avg Module Usage =====")
    print("Module A:", result["avg_module_usage"]["module_a"])
    print("Final Decision:", result["avg_module_usage"]["final_decision"])

    # save_results_json(result, "my_masking.json")