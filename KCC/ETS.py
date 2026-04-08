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
Please answer with Korean.

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
Please answer with Korean.

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
Please answer with Korean.

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
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)
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
# KMMLU Loader Utils
# =========================================================

def normalize_answer_label(answer) -> Optional[str]:
    """
    다양한 정답 표현을 A/B/C/D로 통일
    지원 예:
    - "A", "B", "C", "D"
    - "a", "b", ...
    - 0,1,2,3
    - 1,2,3,4
    """
    if answer is None:
        return None

    if isinstance(answer, str):
        answer = answer.strip()
        if answer.upper() in {"A", "B", "C", "D"}:
            return answer.upper()
        if answer in {"0", "1", "2", "3"}:
            return ["A", "B", "C", "D"][int(answer)]
        if answer in {"1", "2", "3", "4"}:
            return ["A", "B", "C", "D"][int(answer) - 1]

    if isinstance(answer, int):
        if answer in [0, 1, 2, 3]:
            return ["A", "B", "C", "D"][answer]
        if answer in [1, 2, 3, 4]:
            return ["A", "B", "C", "D"][answer - 1]

    raise ValueError(f"Unsupported answer format: {answer}")


def extract_options_from_item(item: dict) -> Dict[str, str]:
    """
    KMMLU/유사 MCQ 포맷을 최대한 유연하게 처리
    지원 예:
    1)
      {"A": "...", "B": "...", "C": "...", "D": "..."}
    2)
      {"options": {"A": "...", "B": "...", "C": "...", "D": "..."}}
    3)
      {"choices": ["...", "...", "...", "..."]}
    4)
      {"choices": {"A": "...", "B": "...", "C": "...", "D": "..."}}
    """
    if all(k in item for k in ["A", "B", "C", "D"]):
        return {
            "A": str(item["A"]),
            "B": str(item["B"]),
            "C": str(item["C"]),
            "D": str(item["D"]),
        }

    if "options" in item:
        opts = item["options"]
        if isinstance(opts, dict) and all(k in opts for k in ["A", "B", "C", "D"]):
            return {
                "A": str(opts["A"]),
                "B": str(opts["B"]),
                "C": str(opts["C"]),
                "D": str(opts["D"]),
            }

    if "choices" in item:
        choices = item["choices"]

        if isinstance(choices, list):
            if len(choices) != 4:
                raise ValueError(f"choices length must be 4, got {len(choices)}")
            return {
                "A": str(choices[0]),
                "B": str(choices[1]),
                "C": str(choices[2]),
                "D": str(choices[3]),
            }

        if isinstance(choices, dict) and all(k in choices for k in ["A", "B", "C", "D"]):
            return {
                "A": str(choices["A"]),
                "B": str(choices["B"]),
                "C": str(choices["C"]),
                "D": str(choices["D"]),
            }

    raise ValueError(f"Cannot extract options from item: {item}")


def extract_question_from_item(item: dict) -> str:
    for key in ["question", "query", "prompt"]:
        if key in item:
            return str(item[key]).strip()
    raise ValueError(f"Cannot find question field in item: {item}")


def extract_answer_from_item(item: dict) -> Optional[str]:
    for key in ["answer", "label", "gold", "target"]:
        if key in item:
            return normalize_answer_label(item[key])
    return None


def convert_to_question_sample(item: dict) -> QuestionSample:
    question = extract_question_from_item(item)
    options = extract_options_from_item(item)
    answer = extract_answer_from_item(item)

    return QuestionSample(
        question=question,
        options=options,
        answer=answer
    )


def load_kmmlu_json(path: str) -> List[QuestionSample]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "data" in data:
            data = data["data"]
        else:
            raise ValueError("JSON root is dict but no 'data' field found.")

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of items or a dict with 'data'.")

    return [convert_to_question_sample(item) for item in data]


def load_kmmlu_jsonl(path: str) -> List[QuestionSample]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            samples.append(convert_to_question_sample(item))
    return samples


def load_kmmlu_hf(
    dataset_name: str = "HAERAE-HUB/KMMLU",
    subject: Optional[str] = None,
    split: str = "test"
) -> List[QuestionSample]:
    """
    Hugging Face datasets에서 직접 불러오기
    예:
        load_kmmlu_hf(dataset_name="HAERAE-HUB/KMMLU", subject="accounting", split="test")
    """
    from datasets import load_dataset

    if subject is None:
        raise ValueError("subject(= config name)를 지정해야 합니다. 예: 'accounting'")

    ds = load_dataset(dataset_name, subject, split=split)

    samples = []
    for item in ds:
        samples.append(convert_to_question_sample(dict(item)))
    return samples


def load_kmmlu(
    path: Optional[str] = None,
    hf_dataset_name: Optional[str] = None,
    hf_subject: Optional[str] = None,
    hf_split: str = "test"
) -> List[QuestionSample]:
    """
    사용 방법:
    1) 로컬 json
       load_kmmlu(path="kmmlu.json")
    2) 로컬 jsonl
       load_kmmlu(path="kmmlu.jsonl")
    3) HF datasets
       load_kmmlu(hf_dataset_name="HAERAE-HUB/KMMLU", hf_subject="accounting", hf_split="test")
    """
    if path is not None:
        if path.endswith(".json"):
            return load_kmmlu_json(path)
        elif path.endswith(".jsonl"):
            return load_kmmlu_jsonl(path)
        else:
            raise ValueError("Only .json and .jsonl are supported for local files.")

    if hf_dataset_name is not None:
        return load_kmmlu_hf(
            dataset_name=hf_dataset_name,
            subject=hf_subject,
            split=hf_split
        )

    raise ValueError("Either path or hf_dataset_name must be provided.")


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
        if chosen_answer_override is not None:
            a_star = chosen_answer_override
        elif len(remaining_options) == 1:
            a_star = remaining_options[0]
        else:
            a_star = max(remaining_options, key=lambda x: confidences[x])

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
        if len(confidences) == 0:
            return list(options.keys())[:1]
        best = max(confidences, key=confidences.get)
        return [best]

    def run(self, sample: QuestionSample, temperature: float = 0.0) -> Dict:
        a_result = self.module_a.run(sample, temperature=temperature)

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
                sub_conf = {opt: confidences[opt] for opt in remaining}
                sorted_opts = sorted(sub_conf, key=sub_conf.get, reverse=True)

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

def evaluate_dataset(
    pipeline: EliminationPipeline,
    dataset: List[QuestionSample],
    temperature: float = 0.0,
    verbose: bool = True
) -> Dict:
    predictions = []
    correct = 0
    total = 0

    for idx, sample in enumerate(dataset):
        output = pipeline.run(sample, temperature=temperature)
        pred = output["module_d"]["final_answer"]

        row = {
            "question": sample.question,
            "prediction": pred,
            "gold": sample.answer,
            "correct": None if sample.answer is None else pred == sample.answer,
            "confidence": output["module_d"]["calibrated_confidence"],
            "final_explanation": output["module_d"]["final_explanation"]
        }
        predictions.append(row)

        if sample.answer is not None:
            total += 1
            correct += int(pred == sample.answer)

        if verbose and (idx + 1) % 10 == 0:
            acc_so_far = correct / total if total > 0 else 0.0
            print(f"[{idx + 1}/{len(dataset)}] current accuracy = {acc_so_far:.4f}")

    acc = (correct / total) if total > 0 else None

    return {
        "accuracy": acc,
        "num_evaluated": total,
        "results": predictions
    }


# =========================================================
# Save Utils
# =========================================================

def save_results_json(result: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# =========================================================
# Example Main
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
        threshold=0.5,   # KMMLU에서는 0.5~0.6 정도 실험 추천
        k_max=2,
        calibration_fn=lambda x: x
    )

    # -----------------------------------------------------
    # 방법 1: 로컬 JSON / JSONL 파일 사용
    # -----------------------------------------------------
    # dataset = load_kmmlu(path="kmmlu_test.json")
    # dataset = load_kmmlu(path="kmmlu_test.jsonl")

    # -----------------------------------------------------
    # 방법 2: Hugging Face datasets 사용
    # subject는 KMMLU의 config 이름으로 넣어야 함
    # 예: "accounting", "law", ...
    # -----------------------------------------------------
    dataset = load_kmmlu(
        hf_dataset_name="HAERAE-HUB/KMMLU",
        hf_subject="accounting",
        hf_split="test"
    )

    print(f"Loaded {len(dataset)} samples")

    result = evaluate_dataset(
        pipeline=pipeline,
        dataset=dataset,
        temperature=0.0,
        verbose=True
    )

    print("\n===== Final Result =====")
    print("Accuracy:", result["accuracy"])
    print("Num evaluated:", result["num_evaluated"])

    save_results_json(result, "kmmlu_results.json")
    print("Saved results to kmmlu_results.json")