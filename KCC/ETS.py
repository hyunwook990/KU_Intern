import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================================
# Data Classes
# =========================================================

@dataclass
class QuestionSample:
    question: str
    options: Dict[str, str]   # {"A": "...", "B": "...", "C": "...", "D": "..."}
    answer: Optional[str] = None


@dataclass
class OptionReasoning:
    decision: str             # "keep" or "eliminate"
    rationale: str
    confidence: float


@dataclass
class ModuleAResult:
    rationales: Dict[str, str]
    confidences: Dict[str, float]
    decisions: Dict[str, str]
    raw_outputs: Dict[str, str]


@dataclass
class ModuleBResult:
    elimination_mask: Dict[str, int]   # 0 keep / 1 eliminate
    remaining_options: List[str]


@dataclass
class ModuleDResult:
    final_answer: str
    final_explanation: str
    calibrated_confidence: float


# =========================================================
# Prompts
# =========================================================

MODULE_A_PROMPT = """You are a careful multiple-choice test taker.
Please answer with Korean

Your task is to evaluate ONLY the target option below.

Question:
{question}

All options:
A. {A}
B. {B}
C. {C}
D. {D}

Target option to evaluate:
{target_option}. {target_text}

Decide whether the target option should be kept as a plausible candidate or eliminated.
Focus on whether this option is consistent with the question compared with the full set of options.

Return ONLY valid JSON in this format:
{{
  "decision": "keep" or "eliminate",
  "rationale": "1-3 sentence explanation",
  "confidence": 0.0
}}

Rules:
- confidence must be a float between 0 and 1
- higher confidence means you are more confident in your decision about this option
- do not return markdown
- do not return any extra text
"""

SELF_DEBATE_PROMPT = """You are a careful multiple-choice test taker.

Question:
{question}

Remaining options:
{opt1}. {text1}
{opt2}. {text2}

Existing rationale:
- {opt1}: {reason1}
- {opt2}: {reason2}

Existing confidence:
- {opt1}: {conf1}
- {opt2}: {conf2}

Compare the two options carefully and decide which one is more likely correct.

Return ONLY valid JSON in this format:
{{
  "winner": "{opt1}" or "{opt2}",
  "reason": "2-4 sentence explanation",
  "confidence": 0.0
}}

Rules:
- confidence must be between 0 and 1
- do not return markdown
- do not return any extra text
"""

MODULE_D_PROMPT = """You are a careful multiple-choice test taker.

Question:
{question}

Options:
A. {A}
B. {B}
C. {C}
D. {D}

Chosen final answer: {chosen}

Reasoning supporting the chosen answer:
{correct_rationale}

Reasoning for rejected or less likely options:
{wrong_rationales}

Write a concise final explanation in 3-5 sentences explaining why the chosen answer is best.
Return only plain text explanation.
"""


# =========================================================
# LLM Wrapper
# =========================================================

class HFLLM:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 300,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map
        )

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
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
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


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
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)  # trailing comma 제거
    return json.loads(raw)


def normalize_decision(decision: str) -> str:
    d = decision.strip().lower()
    if d not in {"keep", "eliminate"}:
        raise ValueError(f"Invalid decision: {decision}")
    return d


def clamp_confidence(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def parse_option_reasoning(raw_output: str) -> OptionReasoning:
    data = safe_json_loads(raw_output)

    decision = normalize_decision(data["decision"])
    rationale = str(data["rationale"]).strip()
    confidence = clamp_confidence(float(data["confidence"]))

    return OptionReasoning(
        decision=decision,
        rationale=rationale,
        confidence=confidence
    )


# =========================================================
# Module A
# =========================================================

class ModuleA:
    """
    각 옵션을 독립적으로 평가해서 rationale/confidence 생성
    """

    def __init__(self, llm: HFLLM):
        self.llm = llm

    def _build_prompt(self, sample: QuestionSample, target_option: str) -> str:
        return MODULE_A_PROMPT.format(
            question=sample.question,
            A=sample.options["A"],
            B=sample.options["B"],
            C=sample.options["C"],
            D=sample.options["D"],
            target_option=target_option,
            target_text=sample.options[target_option]
        )

    def run(self, sample: QuestionSample, temperature: float = 0.0) -> ModuleAResult:
        rationales = {}
        confidences = {}
        decisions = {}
        raw_outputs = {}

        for opt in ["A", "B", "C", "D"]:
            prompt = self._build_prompt(sample, opt)
            raw = self.llm.generate(prompt, temperature=temperature)
            parsed = parse_option_reasoning(raw)

            rationales[opt] = parsed.rationale
            confidences[opt] = parsed.confidence
            decisions[opt] = parsed.decision
            raw_outputs[opt] = raw

        return ModuleAResult(
            rationales=rationales,
            confidences=confidences,
            decisions=decisions,
            raw_outputs=raw_outputs
        )


# =========================================================
# Module B
# =========================================================

class ModuleB:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def run(
        self,
        options: Dict[str, str],
        confidences: Dict[str, float],
        decisions: Optional[Dict[str, str]] = None
    ) -> ModuleBResult:
        """
        기본 규칙:
        - confidence >= threshold 이면 keep
        - confidence < threshold 이면 eliminate

        decisions를 같이 주면,
        decision이 eliminate인데 confidence가 threshold 이상인 이상한 경우를 막기 위해
        decision도 반영할 수 있음.
        """
        elimination_mask = {}
        remaining_options = []

        for opt in options.keys():
            c = confidences[opt]

            keep_by_conf = (c >= self.threshold)
            if decisions is not None:
                keep_by_decision = (decisions[opt] == "keep")
                keep = keep_by_conf and keep_by_decision
            else:
                keep = keep_by_conf

            if keep:
                elimination_mask[opt] = 0
                remaining_options.append(opt)
            else:
                elimination_mask[opt] = 1

        return ModuleBResult(
            elimination_mask=elimination_mask,
            remaining_options=remaining_options
        )


# =========================================================
# Module C
# =========================================================

class ModuleC:
    def __init__(self, k_max: int = 2):
        self.k_max = k_max

    def run(self, remaining_options: List[str], confidences: Dict[str, float], k: int) -> str:
        if len(remaining_options) > 2 and k < self.k_max:
            return "eliminate_more"
        elif len(remaining_options) == 2:
            return "self_debate"
        else:
            return "direct_answer"


# =========================================================
# Self Debate
# =========================================================

def run_self_debate(
    llm: HFLLM,
    sample: QuestionSample,
    remaining_options: List[str],
    rationales: Dict[str, str],
    confidences: Dict[str, float],
    temperature: float = 0.0
) -> Tuple[str, str, float, str]:
    if len(remaining_options) != 2:
        raise ValueError("Self-debate requires exactly 2 remaining options.")

    opt1, opt2 = remaining_options
    prompt = SELF_DEBATE_PROMPT.format(
        question=sample.question,
        opt1=opt1,
        text1=sample.options[opt1],
        opt2=opt2,
        text2=sample.options[opt2],
        reason1=rationales[opt1],
        reason2=rationales[opt2],
        conf1=confidences[opt1],
        conf2=confidences[opt2]
    )

    raw = llm.generate(prompt, temperature=temperature)
    data = safe_json_loads(raw)

    winner = str(data["winner"]).strip().upper()
    reason = str(data["reason"]).strip()
    confidence = clamp_confidence(float(data["confidence"]))

    if winner not in {opt1, opt2}:
        raise ValueError(f"Invalid winner: {winner}")

    return winner, reason, confidence, raw


# =========================================================
# Module D
# =========================================================

class ModuleD:
    def __init__(
        self,
        llm: HFLLM,
        calibration_fn: Optional[Callable[[float], float]] = None
    ):
        self.llm = llm
        self.calibration_fn = calibration_fn if calibration_fn is not None else (lambda x: x)

    def run(
        self,
        sample: QuestionSample,
        remaining_options: List[str],
        rationales: Dict[str, str],
        confidences: Dict[str, float],
        chosen_answer_override: Optional[str] = None,
        temperature: float = 0.0
    ) -> ModuleDResult:
        # 1. final answer selection
        if chosen_answer_override is not None:
            a_star = chosen_answer_override
        elif len(remaining_options) == 1:
            a_star = remaining_options[0]
        else:
            a_star = max(remaining_options, key=lambda x: confidences[x])

        # 2. explanation generation
        correct_rationale = rationales[a_star]
        wrong_rationales = []
        for opt in ["A", "B", "C", "D"]:
            if opt != a_star:
                wrong_rationales.append(f"{opt}: {rationales[opt]}")
        wrong_rationales_text = "\n".join(wrong_rationales)

        prompt = MODULE_D_PROMPT.format(
            question=sample.question,
            A=sample.options["A"],
            B=sample.options["B"],
            C=sample.options["C"],
            D=sample.options["D"],
            chosen=a_star,
            correct_rationale=correct_rationale,
            wrong_rationales=wrong_rationales_text
        )

        final_explanation = self.llm.generate(prompt, temperature=temperature)

        # 3. calibrated confidence
        raw_conf = confidences[a_star]
        calibrated = clamp_confidence(self.calibration_fn(raw_conf))

        return ModuleDResult(
            final_answer=a_star,
            final_explanation=final_explanation,
            calibrated_confidence=calibrated
        )


# =========================================================
# Pipeline
# =========================================================

class EliminationPipeline:
    def __init__(
        self,
        llm: HFLLM,
        threshold: float = 0.5,
        k_max: int = 2,
        calibration_fn: Optional[Callable[[float], float]] = None
    ):
        self.llm = llm
        self.module_a = ModuleA(llm)
        self.module_b = ModuleB(threshold=threshold)
        self.module_c = ModuleC(k_max=k_max)
        self.module_d = ModuleD(llm=llm, calibration_fn=calibration_fn)
        self.threshold = threshold
        self.k_max = k_max

    def _fallback_if_empty(
        self,
        options: Dict[str, str],
        confidences: Dict[str, float]
    ) -> List[str]:
        """
        전부 eliminate 되는 경우, confidence 최고인 옵션 하나는 살림
        """
        if len(confidences) == 0:
            return list(options.keys())[:1]
        best = max(confidences, key=confidences.get)
        return [best]

    def run(self, sample: QuestionSample, temperature: float = 0.0) -> Dict:
        # -------------------------
        # Module A
        # -------------------------
        a_result = self.module_a.run(sample, temperature=temperature)

        # -------------------------
        # Module B
        # -------------------------
        b_result = self.module_b.run(
            options=sample.options,
            confidences=a_result.confidences,
            decisions=a_result.decisions
        )

        remaining = b_result.remaining_options
        if len(remaining) == 0:
            remaining = self._fallback_if_empty(sample.options, a_result.confidences)
            for opt in sample.options:
                b_result.elimination_mask[opt] = 1
            for opt in remaining:
                b_result.elimination_mask[opt] = 0

        confidences = dict(a_result.confidences)
        rationales = dict(a_result.rationales)
        decisions = dict(a_result.decisions)

        k = 0
        chosen_override = None
        self_debate_trace = None

        while True:
            action = self.module_c.run(remaining, confidences, k)

            if action == "eliminate_more":
                # 남은 옵션만 재평가
                sub_conf = {opt: confidences[opt] for opt in remaining}
                sorted_opts = sorted(sub_conf, key=sub_conf.get, reverse=True)

                # 상위 2개만 남기는 간단한 제거 전략
                remaining = sorted_opts[:2]
                k += 1

            elif action == "self_debate":
                winner, reason, debate_conf, raw = run_self_debate(
                    llm=self.llm,
                    sample=sample,
                    remaining_options=remaining,
                    rationales=rationales,
                    confidences=confidences,
                    temperature=temperature
                )
                chosen_override = winner
                confidences[winner] = max(confidences[winner], debate_conf)
                self_debate_trace = {
                    "winner": winner,
                    "reason": reason,
                    "confidence": debate_conf,
                    "raw_output": raw
                }
                break

            else:
                break

        # -------------------------
        # Module D
        # -------------------------
        d_result = self.module_d.run(
            sample=sample,
            remaining_options=remaining,
            rationales=rationales,
            confidences=confidences,
            chosen_answer_override=chosen_override,
            temperature=temperature
        )

        return {
            "module_a": {
                "decisions": decisions,
                "rationales": rationales,
                "confidences": confidences,
                "raw_outputs": a_result.raw_outputs
            },
            "module_b": {
                "elimination_mask": b_result.elimination_mask,
                "remaining_options": b_result.remaining_options
            },
            "module_c": {
                "final_remaining_options": remaining,
                "self_debate": self_debate_trace
            },
            "module_d": {
                "final_answer": d_result.final_answer,
                "final_explanation": d_result.final_explanation,
                "calibrated_confidence": d_result.calibrated_confidence
            }
        }


# =========================================================
# Evaluation
# =========================================================

def evaluate_dataset(pipeline: EliminationPipeline, dataset: List[QuestionSample]) -> Dict:
    predictions = []
    correct = 0
    total = 0

    for sample in dataset:
        output = pipeline.run(sample)
        pred = output["module_d"]["final_answer"]

        row = {
            "question": sample.question,
            "prediction": pred,
            "gold": sample.answer,
            "correct": None if sample.answer is None else pred == sample.answer,
            "confidence": output["module_d"]["calibrated_confidence"]
        }
        predictions.append(row)

        if sample.answer is not None:
            total += 1
            correct += int(pred == sample.answer)

    acc = (correct / total) if total > 0 else None

    return {
        "accuracy": acc,
        "num_evaluated": total,
        "results": predictions
    }


# =========================================================
# Example
# =========================================================

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    llm = HFLLM(
        model_name=model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        max_new_tokens=300
    )

    pipeline = EliminationPipeline(
        llm=llm,
        threshold=0.5,
        k_max=2,
        calibration_fn=lambda x: x
    )

    sample = QuestionSample(
        question="한국채택국제회계기준(K-IFRS)하에서 금융자산으로 분류되지 않는 것은?",
        options={
            "A": "대여금",
            "B": "재고자산",
            "C": "매출채권",
            "D": "만기보유금융자산"
        },
    )

    result = pipeline.run(sample)
    print(json.dumps(result, ensure_ascii=False, indent=2))