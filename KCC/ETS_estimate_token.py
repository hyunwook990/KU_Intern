import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


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
class ModuleBResult:
    elimination_mask: List[int]         # 0: keep, 1: eliminate
    remaining_indices: List[int]        # local indices


@dataclass
class ModuleCResult:
    next_action: str                    # "eliminate_more" / "self_debate" / "direct_answer"
    top_confidence: Optional[float] = None
    second_confidence: Optional[float] = None
    confidence_gap: Optional[float] = None
    reason: Optional[str] = None


@dataclass
class SelfDebateResult:
    winner_local_index: int
    reason: str
    confidence: float
    raw_output: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    num_calls: int = 1


@dataclass
class ModuleDResult:
    final_answer_label: str
    final_explanation: str
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
    eliminated_round: int


# =========================================================
# Prompt Templates
# =========================================================

# -------------------------------
# Module A: baseline faithful version
# Q + O_i only
# -------------------------------
MODULE_A_PROMPT_KO = """당신은 객관식 문제의 하나의 선택지만 평가하는 전문가입니다.

문제:
{question}

선택지:
{target_label}. {target_option}

작업:
1. 이 선택지가 정답인지 아닌지 판단하세요.
2. 1~3문장으로 이유를 설명하세요.
3. 이 선택지가 틀릴 수 있는 이유 또는 한계를 반드시 포함하세요.

반드시 아래 JSON 형식으로만 답하세요:
{
  "rationale": "설명",
  "confidence": 0.0
}

규칙:
- 이 선택지가 맞다고 가정하지 마세요.
- 이 선택지 자체만 보고 판단하세요.
- confidence는 이 선택지가 정답일 확률을 의미합니다.
- JSON 이외의 텍스트는 출력하지 마세요.
"""

MODULE_A_PROMPT = """You are evaluating a single answer option in a multiple-choice question.
Please answer in Korean.

Question:
{question}

Option:
{target_label}. {target_option}

Task:
1. Determine whether this option is correct or incorrect.
2. Explain your reasoning in 1-3 sentences.
3. Explicitly identify why this option could be wrong or what limitation it has.

Return ONLY valid JSON:
{
  "rationale": "your explanation",
  "confidence": 0.0
}

Rules:
- Do NOT assume the option is correct.
- Evaluate only this option itself.
- Confidence must represent the probability that THIS option is correct.
- Do not return markdown or extra text.
"""

SELF_DEBATE_PROMPT_KO = """당신은 두 개의 선택지를 비교하는 전문가입니다.

문제:
{question}

선택지 1:
{opt1_label}. {opt1_text}

선택지 2:
{opt2_label}. {opt2_text}

기존 설명:
- {opt1_label}: {reason1}
- {opt2_label}: {reason2}

기존 confidence:
- {opt1_label}: {conf1}
- {opt2_label}: {conf2}

작업:
1. 두 선택지를 비교하세요.
2. 어떤 선택지가 더 정답에 가까운지 판단하세요.
3. 두 선택지의 핵심 차이를 설명하세요.
4. 탈락하는 선택지가 왜 더 부적절한지 명확히 설명하세요.

반드시 아래 JSON 형식으로만 답하세요:
{
  "winner": "{opt1_label}" 또는 "{opt2_label}",
  "reason": "설명",
  "confidence": 0.0
}

규칙:
- 반드시 탈락하는 선택지의 한계나 문제점을 명확히 설명하세요.
- 모호한 설명은 금지합니다.
- confidence는 판단에 대한 확신도를 의미합니다.
- JSON 이외의 텍스트는 출력하지 마세요.
"""

SELF_DEBATE_PROMPT = """You are comparing two candidate answers.
Please answer in Korean.

Question:
{question}

Option 1:
{opt1_label}. {opt1_text}

Option 2:
{opt2_label}. {opt2_text}

Existing rationale:
- {opt1_label}: {reason1}
- {opt2_label}: {reason2}

Existing confidence:
- {opt1_label}: {conf1}
- {opt2_label}: {conf2}

Task:
1. Compare the two options carefully.
2. Identify the key difference between them.
3. Clearly explain why one is more appropriate.
4. Explicitly state the limitation of the losing option.

Return ONLY valid JSON:
{
  "winner": "{opt1_label}" or "{opt2_label}",
  "reason": "your explanation",
  "confidence": 0.0
}

Rules:
- You MUST explain why the losing option is less appropriate.
- Avoid vague reasoning.
- Confidence must reflect how certain the decision is.
- Do not return markdown or extra text.
"""

# -------------------------------
# Module D: summarize prior reasoning
# -------------------------------
MODULE_D_PROMPT_KO = """당신은 객관식 문제 풀이 결과를 최종 정리하는 전문가입니다.

문제:
{question}

최종 선택:
{chosen_label}. {chosen_text}

선택된 이유:
{chosen_rationale}

제거된 선택지들에 대한 기존 판단:
{eliminated_rationales_text}

아직 남아 있었지만 최종 선택되지 않은 선택지들에 대한 기존 판단:
{other_remaining_rationales_text}

작업:
1. 최종 선택이 왜 가장 적절한지 간단히 정리하세요.
2. 제거되었거나 최종 선택되지 않은 선택지들이 왜 제외되었는지, 앞서 주어진 판단을 바탕으로 간단히 요약하세요.
3. 전체 설명은 2~4문장으로 작성하세요.

규칙:
- 새로운 판단을 추가로 만들기보다, 앞 단계의 reasoning을 일관되게 정리하세요.
- 최종 설명은 정답의 타당성과 다른 선택지가 제외된 이유를 모두 포함해야 합니다.
- JSON이나 마크다운 없이 일반 텍스트로만 답하세요.
"""

MODULE_D_PROMPT = """You are finalizing the result of a multiple-choice reasoning process.
Please answer in Korean.

Question:
{question}

Chosen final answer:
{chosen_label}. {chosen_text}

Reasoning for the chosen answer:
{chosen_rationale}

Previously derived reasons for eliminated options:
{eliminated_rationales_text}

Previously derived reasons for options that remained but were not finally chosen:
{other_remaining_rationales_text}

Task:
1. Briefly summarize why the chosen answer is the most appropriate.
2. Briefly summarize why the other options were excluded, based on the earlier reasoning.
3. Keep the explanation concise (2-4 sentences).

Return ONLY plain text explanation.

Rules:
- Do not re-solve the question from scratch.
- Do not invent a completely new judgment; summarize the earlier reasoning consistently.
- The final explanation must include both the justification for the chosen answer and why the other options were excluded.
- Do not return JSON or markdown.
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


# =========================================================
# 데이터셋 로드
# =========================================================

def convert_item_to_sample(item: dict) -> QuestionSample:
    options = []
    for key in ["A", "B", "C", "D"]:
        if key in item:
            options.append(str(item[key]).strip())

    if not options:
        raise ValueError("선택지를 찾을 수 없습니다.")

    return QuestionSample(
        question=str(item["question"]).strip(),
        options=options,
        answer=answer_to_label(item.get("answer"))
    )


def load_kmmlu_dataset(
    dataset_name: str = "HAERAE-HUB/KMMLU",
    subject: str = "Accounting",
    split: str = "dev",
    max_samples: Optional[int] = None
) -> List[QuestionSample]:
    ds = load_dataset(dataset_name, subject, split=split)

    samples = []
    for i, item in enumerate(ds):
        try:
            samples.append(convert_item_to_sample(dict(item)))
        except Exception as e:
            print(f"[Skip] index={i}, reason={e}")

        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples


# =========================================================
# Module A
# baseline: Q + O_i only
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
            parsed = parse_module_a_option_output(
                gen_result.text,
                gen_result.input_tokens,
                gen_result.output_tokens,
                gen_result.total_tokens
            )

            rationales.append(parsed.rationale)
            confidences.append(parsed.confidence)
            raw_outputs.append(parsed.raw_output)

            input_tokens_list.append(parsed.input_tokens)
            output_tokens_list.append(parsed.output_tokens)
            total_tokens_list.append(parsed.total_tokens)

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
# Module B
# =========================================================

class ModuleB:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def run(
        self,
        options: List[str],
        confidences: List[float],
        threshold: Optional[float] = None
    ) -> ModuleBResult:
        actual_threshold = self.threshold if threshold is None else threshold

        print("==============================Module_B==============================")

        if len(options) != len(confidences):
            raise ValueError(
                f"options와 confidences 길이가 다릅니다: "
                f"len(options)={len(options)}, len(confidences)={len(confidences)}"
            )

        elimination_mask = []
        remaining_indices = []

        for idx, conf in enumerate(confidences):
            if conf < actual_threshold:
                elimination_mask.append(1)
            else:
                elimination_mask.append(0)
                remaining_indices.append(idx)

        return ModuleBResult(
            elimination_mask=elimination_mask,
            remaining_indices=remaining_indices
        )


# =========================================================
# Module C
# =========================================================

class ModuleC:
    def __init__(
        self,
        max_count: int = 2,
        tau_answer: float = 0.9,
        debate_gap_threshold: float = 0.05
    ):
        self.max_count = max_count
        self.tau_answer = tau_answer
        self.debate_gap_threshold = debate_gap_threshold

    def run(
        self,
        remaining_indices: List[int],
        remaining_confidences: List[float],
        count: int
    ) -> ModuleCResult:
        print("==============================Module_C==============================")
        num_remaining = len(remaining_indices)

        if num_remaining == 0:
            return ModuleCResult(
                next_action="direct_answer",
                reason="no_remaining_options_fallback"
            )

        sorted_conf = sorted(remaining_confidences, reverse=True)
        top_conf = sorted_conf[0] if len(sorted_conf) >= 1 else None
        second_conf = sorted_conf[1] if len(sorted_conf) >= 2 else None
        gap = None if second_conf is None else top_conf - second_conf

        if num_remaining == 1:
            return ModuleCResult(
                next_action="direct_answer",
                top_confidence=top_conf,
                second_confidence=second_conf,
                confidence_gap=gap,
                reason="single_option_left"
            )

        if count >= self.max_count:
            return ModuleCResult(
                next_action="direct_answer",
                top_confidence=top_conf,
                second_confidence=second_conf,
                confidence_gap=gap,
                reason="max_count_reached"
            )

        if top_conf is not None and top_conf >= self.tau_answer:
            return ModuleCResult(
                next_action="direct_answer",
                top_confidence=top_conf,
                second_confidence=second_conf,
                confidence_gap=gap,
                reason="top_confidence_above_tau_answer"
            )

        if num_remaining == 2:
            return ModuleCResult(
                next_action="self_debate",
                top_confidence=top_conf,
                second_confidence=second_conf,
                confidence_gap=gap,
                reason="exactly_two_options_remain"
            )

        if gap is not None and gap <= self.debate_gap_threshold:
            return ModuleCResult(
                next_action="self_debate",
                top_confidence=top_conf,
                second_confidence=second_conf,
                confidence_gap=gap,
                reason="top_two_confidences_are_close"
            )

        return ModuleCResult(
            next_action="eliminate_more",
            top_confidence=top_conf,
            second_confidence=second_conf,
            confidence_gap=gap,
            reason="need_more_elimination"
        )


# =========================================================
# Self Debate
# =========================================================

def run_self_debate(
    llm: HFLLM,
    question: str,
    options: List[str],
    labels: List[str],
    rationales: List[str],
    confidences: List[float],
    temperature: float = 0.0
) -> SelfDebateResult:
    print("==============================Self_Debate==============================")
    if len(options) != 2:
        raise ValueError("Self-debate는 선택지가 정확히 2개일 때만 실행됩니다.")

    prompt = SELF_DEBATE_PROMPT.format(
        question=question,
        opt1_label=labels[0],
        opt1_text=options[0],
        opt2_label=labels[1],
        opt2_text=options[1],
        reason1=rationales[0],
        reason2=rationales[1],
        conf1=confidences[0],
        conf2=confidences[1]
    )

    gen_result = llm.generate(prompt, temperature=temperature)
    data = safe_json_loads(gen_result.text)

    winner_label = str(data["winner"]).strip().upper()
    reason = str(data["reason"]).strip()
    confidence = clamp_confidence(float(data["confidence"]))

    if winner_label not in labels:
        raise ValueError(f"Invalid winner: {winner_label}, valid={labels}")

    winner_local_index = labels.index(winner_label)

    return SelfDebateResult(
        winner_local_index=winner_local_index,
        reason=reason,
        confidence=confidence,
        raw_output=gen_result.text,
        input_tokens=gen_result.input_tokens,
        output_tokens=gen_result.output_tokens,
        total_tokens=gen_result.total_tokens
    )


# =========================================================
# Module D
# summarize prior reasoning + calibrate confidence
# =========================================================

class ModuleD:
    def __init__(
        self,
        llm: HFLLM,
        calibration_fn: Optional[Callable[[float], float]] = None
    ):
        self.llm = llm
        self.calibration_fn = calibration_fn if calibration_fn is not None else (lambda x: x)

    def _argmax_index(self, values: List[float]) -> int:
        return max(range(len(values)), key=lambda i: values[i])

    def run(
        self,
        question: str,
        remaining_labels: List[str],
        remaining_options: List[str],
        rationales: List[str],
        confidences: List[float],
        eliminated_records: List[EliminatedOptionRecord],
        temperature: float = 0.0
    ) -> ModuleDResult:
        print("==============================Module_D==============================")

        if not (
            len(remaining_labels) == len(remaining_options) ==
            len(rationales) == len(confidences)
        ):
            raise ValueError(
                "remaining_labels, remaining_options, rationales, confidences의 길이는 같아야 합니다."
            )

        if len(remaining_labels) == 0:
            raise ValueError("Module D에 전달된 후보가 없습니다.")

        if len(remaining_labels) == 1:
            chosen_idx = 0
        else:
            chosen_idx = self._argmax_index(confidences)

        chosen_label = remaining_labels[chosen_idx]
        chosen_text = remaining_options[chosen_idx]
        chosen_rationale = rationales[chosen_idx]
        chosen_confidence = confidences[chosen_idx]

        eliminated_labels = [r.label for r in eliminated_records]
        eliminated_options = [r.option_text for r in eliminated_records]
        eliminated_rationales = [r.rationale for r in eliminated_records]
        eliminated_confidences = [r.confidence for r in eliminated_records]

        eliminated_rationales_text = format_reason_block(
            labels=eliminated_labels,
            options=eliminated_options,
            rationales=eliminated_rationales,
            confidences=eliminated_confidences
        )

        other_remaining_labels = []
        other_remaining_options = []
        other_remaining_rationales = []
        other_remaining_confidences = []

        for i in range(len(remaining_labels)):
            if i == chosen_idx:
                continue
            other_remaining_labels.append(remaining_labels[i])
            other_remaining_options.append(remaining_options[i])
            other_remaining_rationales.append(rationales[i])
            other_remaining_confidences.append(confidences[i])

        other_remaining_rationales_text = format_reason_block(
            labels=other_remaining_labels,
            options=other_remaining_options,
            rationales=other_remaining_rationales,
            confidences=other_remaining_confidences
        )

        prompt = MODULE_D_PROMPT.format(
            question=question,
            chosen_label=chosen_label,
            chosen_text=chosen_text,
            chosen_rationale=chosen_rationale,
            eliminated_rationales_text=eliminated_rationales_text,
            other_remaining_rationales_text=other_remaining_rationales_text
        )

        gen_result = self.llm.generate(prompt, temperature=temperature)
        calibrated_confidence = clamp_confidence(self.calibration_fn(chosen_confidence))

        return ModuleDResult(
            final_answer_label=chosen_label,
            final_explanation=gen_result.text.strip(),
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
        threshold: float = 0.5,
        max_count: int = 2,
        tau_answer: float = 0.9,
        debate_gap_threshold: float = 0.05,
        calibration_fn: Optional[Callable[[float], float]] = None
    ):
        self.llm = llm
        self.module_a_prompt_template = module_a_prompt_template
        self.module_a = ModuleA(llm)
        self.module_b = ModuleB(threshold=threshold)
        self.module_c = ModuleC(
            max_count=max_count,
            tau_answer=tau_answer,
            debate_gap_threshold=debate_gap_threshold
        )
        self.module_d = ModuleD(llm=llm, calibration_fn=calibration_fn)

    def _subset_by_indices(self, values: List, indices: List[int]) -> List:
        return [values[i] for i in indices]

    def _labels_from_global_indices(self, global_indices: List[int]) -> List[str]:
        return [option_label(i) for i in global_indices]

    def _select_top2_local_indices(self, confidences: List[float]) -> List[int]:
        if len(confidences) < 2:
            raise ValueError("top-2를 선택하려면 최소 2개의 confidence가 필요합니다.")
        return sorted(
            range(len(confidences)),
            key=lambda i: confidences[i],
            reverse=True
        )[:2]

    def _collect_newly_eliminated_records(
        self,
        current_options: List[str],
        current_global_indices: List[int],
        rationales: List[str],
        confidences: List[float],
        elimination_mask: List[int],
        count: int
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
                    eliminated_round=count
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
                "self_debate": {
                    "num_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                },
                "module_d": {
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

    def run(
        self,
        sample: QuestionSample,
        prompt_template: Optional[str] = None,
        temperature: float = 0.0
    ) -> Dict:
        actual_prompt_template = prompt_template or self.module_a_prompt_template

        current_options = list(sample.options)
        current_global_indices = list(range(len(sample.options)))
        count = 0

        trace = []
        eliminated_records: List[EliminatedOptionRecord] = []
        sample_usage = self._init_usage()

        while True:
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

            b_result = self.module_b.run(
                options=current_options,
                confidences=a_result.confidences
            )

            newly_eliminated = self._collect_newly_eliminated_records(
                current_options=current_options,
                current_global_indices=current_global_indices,
                rationales=a_result.rationales,
                confidences=a_result.confidences,
                elimination_mask=b_result.elimination_mask,
                count=count
            )
            eliminated_records.extend(newly_eliminated)

            remaining_local_indices = b_result.remaining_indices
            remaining_global_indices = self._subset_by_indices(current_global_indices, remaining_local_indices)
            remaining_labels = self._labels_from_global_indices(remaining_global_indices)
            remaining_options = self._subset_by_indices(current_options, remaining_local_indices)
            remaining_confidences = self._subset_by_indices(a_result.confidences, remaining_local_indices)
            remaining_rationales = self._subset_by_indices(a_result.rationales, remaining_local_indices)

            c_result = self.module_c.run(
                remaining_indices=remaining_global_indices,
                remaining_confidences=remaining_confidences,
                count=count
            )

            trace.append({
                "count": count,
                "current_options": current_options,
                "current_global_indices": current_global_indices,
                "module_a": {
                    "rationales": a_result.rationales,
                    "confidences": a_result.confidences,
                    "raw_outputs": a_result.raw_outputs,
                    "input_tokens_list": a_result.input_tokens_list,
                    "output_tokens_list": a_result.output_tokens_list,
                    "total_tokens_list": a_result.total_tokens_list,
                    "num_calls": a_result.num_calls,
                    "total_input_tokens": a_result.total_input_tokens,
                    "total_output_tokens": a_result.total_output_tokens,
                    "total_tokens": a_result.total_tokens
                },
                "module_b": {
                    "elimination_mask": b_result.elimination_mask,
                    "remaining_local_indices": remaining_local_indices,
                    "remaining_global_indices": remaining_global_indices,
                    "remaining_labels": remaining_labels,
                    "newly_eliminated": [
                        {
                            "label": r.label,
                            "option_text": r.option_text,
                            "rationale": r.rationale,
                            "confidence": r.confidence,
                            "eliminated_round": r.eliminated_round
                        } for r in newly_eliminated
                    ]
                },
                "module_c": {
                    "next_action": c_result.next_action,
                    "top_confidence": c_result.top_confidence,
                    "second_confidence": c_result.second_confidence,
                    "confidence_gap": c_result.confidence_gap,
                    "reason": c_result.reason
                },
                "running_usage": {
                    "num_calls": sample_usage["num_calls"],
                    "total_input_tokens": sample_usage["total_input_tokens"],
                    "total_output_tokens": sample_usage["total_output_tokens"],
                    "total_tokens": sample_usage["total_tokens"],
                    "by_module": {
                        "module_a": dict(sample_usage["by_module"]["module_a"]),
                        "self_debate": dict(sample_usage["by_module"]["self_debate"]),
                        "module_d": dict(sample_usage["by_module"]["module_d"])
                    }
                }
            })

            if len(remaining_local_indices) == 0:
                fallback_local_idx = max(range(len(a_result.confidences)), key=lambda i: a_result.confidences[i])
                fallback_global_index = current_global_indices[fallback_local_idx]
                fallback_label = option_label(fallback_global_index)
                fallback_option = current_options[fallback_local_idx]
                fallback_rationale = a_result.rationales[fallback_local_idx]
                fallback_confidence = a_result.confidences[fallback_local_idx]

                trace[-1]["fallback"] = {
                    "fallback_local_idx": fallback_local_idx,
                    "fallback_global_index": fallback_global_index,
                    "fallback_label": fallback_label,
                    "fallback_confidence": fallback_confidence
                }

                d_eliminated_records = [
                    r for r in eliminated_records
                    if r.global_index != fallback_global_index
                ]

                d_result = self.module_d.run(
                    question=sample.question,
                    remaining_labels=[fallback_label],
                    remaining_options=[fallback_option],
                    rationales=[fallback_rationale],
                    confidences=[fallback_confidence],
                    eliminated_records=d_eliminated_records,
                    temperature=temperature
                )

                self._add_module_usage(
                    usage=sample_usage,
                    module_name="module_d",
                    num_calls=d_result.num_calls,
                    input_tokens=d_result.input_tokens,
                    output_tokens=d_result.output_tokens,
                    total_tokens=d_result.total_tokens
                )

                return {
                    "trace": trace,
                    "final": {
                        "answer_label": d_result.final_answer_label,
                        "answer_text": sample.options[label_to_index(d_result.final_answer_label)],
                        "final_explanation": d_result.final_explanation,
                        "confidence": d_result.calibrated_confidence
                    },
                    "usage": sample_usage
                }

            if c_result.next_action == "eliminate_more":
                current_options = remaining_options
                current_global_indices = remaining_global_indices
                count += 1
                continue

            if c_result.next_action == "self_debate":
                if len(remaining_options) == 2:
                    debate_local_indices = [0, 1]
                else:
                    debate_local_indices = self._select_top2_local_indices(remaining_confidences)

                debate_options = [remaining_options[i] for i in debate_local_indices]
                debate_labels = [remaining_labels[i] for i in debate_local_indices]
                debate_rationales = [remaining_rationales[i] for i in debate_local_indices]
                debate_confidences = [remaining_confidences[i] for i in debate_local_indices]
                debate_global_indices = [remaining_global_indices[i] for i in debate_local_indices]

                debate_result = run_self_debate(
                    llm=self.llm,
                    question=sample.question,
                    options=debate_options,
                    labels=debate_labels,
                    rationales=debate_rationales,
                    confidences=debate_confidences,
                    temperature=temperature
                )

                self._add_module_usage(
                    usage=sample_usage,
                    module_name="self_debate",
                    num_calls=debate_result.num_calls,
                    input_tokens=debate_result.input_tokens,
                    output_tokens=debate_result.output_tokens,
                    total_tokens=debate_result.total_tokens
                )

                winner_local_idx = debate_result.winner_local_index
                loser_local_idx = 1 - winner_local_idx if len(debate_local_indices) == 2 else None

                trace[-1]["self_debate"] = {
                    "candidate_local_indices": debate_local_indices,
                    "candidate_labels": debate_labels,
                    "candidate_options": debate_options,
                    "candidate_rationales": debate_rationales,
                    "candidate_confidences": debate_confidences,
                    "winner_label": debate_labels[winner_local_idx],
                    "winner_option": debate_options[winner_local_idx],
                    "reason": debate_result.reason,
                    "confidence": debate_result.confidence,
                    "raw_output": debate_result.raw_output,
                    "input_tokens": debate_result.input_tokens,
                    "output_tokens": debate_result.output_tokens,
                    "total_tokens": debate_result.total_tokens
                }

                d_eliminated_records = list(eliminated_records)
                if loser_local_idx is not None:
                    d_eliminated_records.append(
                        EliminatedOptionRecord(
                            global_index=debate_global_indices[loser_local_idx],
                            label=debate_labels[loser_local_idx],
                            option_text=debate_options[loser_local_idx],
                            rationale=debate_rationales[loser_local_idx],
                            confidence=debate_confidences[loser_local_idx],
                            eliminated_round=count
                        )
                    )

                d_result = self.module_d.run(
                    question=sample.question,
                    remaining_labels=[debate_labels[winner_local_idx]],
                    remaining_options=[debate_options[winner_local_idx]],
                    rationales=[debate_result.reason],
                    confidences=[max(debate_confidences[winner_local_idx], debate_result.confidence)],
                    eliminated_records=d_eliminated_records,
                    temperature=temperature
                )

                self._add_module_usage(
                    usage=sample_usage,
                    module_name="module_d",
                    num_calls=d_result.num_calls,
                    input_tokens=d_result.input_tokens,
                    output_tokens=d_result.output_tokens,
                    total_tokens=d_result.total_tokens
                )

                return {
                    "trace": trace,
                    "final": {
                        "answer_label": d_result.final_answer_label,
                        "answer_text": sample.options[label_to_index(d_result.final_answer_label)],
                        "final_explanation": d_result.final_explanation,
                        "confidence": d_result.calibrated_confidence
                    },
                    "usage": sample_usage
                }

            d_result = self.module_d.run(
                question=sample.question,
                remaining_labels=remaining_labels,
                remaining_options=remaining_options,
                rationales=remaining_rationales,
                confidences=remaining_confidences,
                eliminated_records=eliminated_records,
                temperature=temperature
            )

            self._add_module_usage(
                usage=sample_usage,
                module_name="module_d",
                num_calls=d_result.num_calls,
                input_tokens=d_result.input_tokens,
                output_tokens=d_result.output_tokens,
                total_tokens=d_result.total_tokens
            )

            return {
                "trace": trace,
                "final": {
                    "answer_label": d_result.final_answer_label,
                    "answer_text": sample.options[label_to_index(d_result.final_answer_label)],
                    "final_explanation": d_result.final_explanation,
                    "confidence": d_result.calibrated_confidence
                },
                "usage": sample_usage
            }


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
        "self_debate": {
            "num_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        },
        "module_d": {
            "num_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
    }


def update_dataset_module_usage(dataset_module_usage: Dict, sample_module_usage: Dict) -> None:
    for module_name in ["module_a", "self_debate", "module_d"]:
        dataset_module_usage[module_name]["num_calls"] += sample_module_usage[module_name]["num_calls"]
        dataset_module_usage[module_name]["input_tokens"] += sample_module_usage[module_name]["input_tokens"]
        dataset_module_usage[module_name]["output_tokens"] += sample_module_usage[module_name]["output_tokens"]
        dataset_module_usage[module_name]["total_tokens"] += sample_module_usage[module_name]["total_tokens"]


def compute_avg_module_usage(dataset_module_usage: Dict, num_processed: int) -> Dict:
    if num_processed == 0:
        return {
            "module_a": {
                "avg_num_calls": None,
                "avg_input_tokens": None,
                "avg_output_tokens": None,
                "avg_total_tokens": None
            },
            "self_debate": {
                "avg_num_calls": None,
                "avg_input_tokens": None,
                "avg_output_tokens": None,
                "avg_total_tokens": None
            },
            "module_d": {
                "avg_num_calls": None,
                "avg_input_tokens": None,
                "avg_output_tokens": None,
                "avg_total_tokens": None
            }
        }

    avg_usage = {}
    for module_name in ["module_a", "self_debate", "module_d"]:
        avg_usage[module_name] = {
            "avg_num_calls": dataset_module_usage[module_name]["num_calls"] / num_processed,
            "avg_input_tokens": dataset_module_usage[module_name]["input_tokens"] / num_processed,
            "avg_output_tokens": dataset_module_usage[module_name]["output_tokens"] / num_processed,
            "avg_total_tokens": dataset_module_usage[module_name]["total_tokens"] / num_processed
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
                print("prediction:", pred_label)
                print("gold:", sample.answer)
                print("num_calls:", usage["num_calls"])
                print("total_input_tokens:", usage["total_input_tokens"])
                print("total_output_tokens:", usage["total_output_tokens"])
                print("total_tokens:", usage["total_tokens"])
                print("module_a_total_tokens:", usage["by_module"]["module_a"]["total_tokens"])
                print("self_debate_total_tokens:", usage["by_module"]["self_debate"]["total_tokens"])
                print("module_d_total_tokens:", usage["by_module"]["module_d"]["total_tokens"])
                print("#############################################################")
                print(f"[{idx + 1}/{len(dataset)}] current accuracy = {acc_so_far:.4f}, skipped = {skipped}")

        except Exception as e:
            skipped += 1
            print(f"[Error] sample index={idx}, reason={e}")

    acc = (correct / total) if total > 0 else None
    num_processed = len(predictions)

    avg_input_tokens = dataset_total_input_tokens / num_processed if num_processed > 0 else None
    avg_output_tokens = dataset_total_output_tokens / num_processed if num_processed > 0 else None
    avg_total_tokens = dataset_total_tokens / num_processed if num_processed > 0 else None
    avg_num_calls = dataset_total_calls / num_processed if num_processed > 0 else None
    avg_module_usage = compute_avg_module_usage(dataset_module_usage, num_processed)

    return {
        "accuracy": acc,
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
        "dataset_module_usage": dataset_module_usage,
        "avg_module_usage": avg_module_usage,
        "results": predictions
    }


def save_results_json(result: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    llm = HFLLM(
        model_name=MODEL_NAME,
        device_map="auto",
        max_new_tokens=300
    )

    pipeline = EliminationPipeline(
        llm=llm,
        module_a_prompt_template=MODULE_A_PROMPT,
        threshold=0.5,
        max_count=2,
        tau_answer=0.9,
        debate_gap_threshold=0.05,
        calibration_fn=lambda x: x
    )

    dataset = load_kmmlu_dataset(
        dataset_name="HAERAE-HUB/KMMLU",
        subject="Accounting",
        split="test",
        max_samples=100
    )

    print(f"Loaded {len(dataset)} samples")

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
    print("Self Debate:", result["dataset_module_usage"]["self_debate"])
    print("Module D:", result["dataset_module_usage"]["module_d"])

    print("\n===== Avg Module Usage =====")
    print("Module A:", result["avg_module_usage"]["module_a"])
    print("Self Debate:", result["avg_module_usage"]["self_debate"])
    print("Module D:", result["avg_module_usage"]["module_d"])

    save_results_json(result, "ets_accounting_test.json")
    print("Saved results to ets_accounting_test.json")