import re
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------
# 1. 설정
# -------------------------------
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)


# -------------------------------
# 2. 데이터 구조
# -------------------------------
@dataclass
class MCQSample:
    question: str
    options: List[str]
    context: Optional[str] = None
    answer: Optional[int] = None  # 정답 인덱스 (선택사항)


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
# 4. 프롬프트 구성
# -------------------------------
def format_options(option_indices: List[int], all_options: List[str]) -> str:
    """
    현재 살아남은 옵션만 보기 좋게 문자열로 변환
    예:
    A. ...
    C. ...
    D. ...
    """
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
    """
    module_1의 Step 1: Prompt Construction
    """
    option_text = format_options(active_indices, options)

    context_block = f"Context:\n{context}\n\n" if context else ""

    prompt = f"""
You are solving a multiple-choice question by process of elimination.
Please answer with Korean

Your task:
1. Read the question carefully.
2. Consider only the currently remaining options.
3. Briefly explain why some options are less plausible.
4. Choose exactly ONE option to eliminate because it is the least plausible.
5. Use the following format exactly:

Reasoning: <your brief reasoning>
Eliminate: <OPTION_LABEL>

Example:
Reasoning: B is not correct because ..
Eliminate: B

{context_block}Question:
{question}

Current remaining options:
{option_text}
""".strip()

    return prompt


# -------------------------------
# 5. 생성 함수
# -------------------------------
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9
) -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

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
# 6. 제거할 선택지 파싱
# -------------------------------
def parse_elimination_decision(
    response_text: str,
    active_indices: List[int]
) -> Optional[int]:
    """
    module_1의 Step 3: Parse Elimination Decision

    가능한 출력 예:
    - Eliminate: B
    - eliminate option C
    - Final decision: remove A
    """

    # 허용되는 라벨만 추출
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

    # 마지막 fallback: 텍스트 안에 단독 알파벳(A/B/C/D...)가 있는지 확인
    candidates = re.findall(r"\b([A-Z])\b", response_text.upper())
    for c in reversed(candidates):
        if c in valid_labels:
            return valid_labels[c]

    return None


# -------------------------------
# 7. module_1 구현
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
    """
    Require: Question q, Context c, Current option set Ot (|Ot| > 1)
    Ensure: Revised option set Ot+1 after eliminating one option
    """

    # Step 1: Prompt Construction
    prompt = build_elimination_prompt(
        question=question,
        options=options,
        active_indices=current_option_set,
        context=context
    )

    # Step 2: Model Reasoning
    reasoning = generate_text(model, tokenizer, prompt)

    # Step 3: Parse Elimination Decision
    eliminated_idx = parse_elimination_decision(reasoning, current_option_set)

    if eliminated_idx is None:
        # 파싱 실패 시 fallback: 마지막 옵션 제거 대신,
        # 가장 뒤의 옵션을 임시 제거하는 방식보다
        # 안전하게 예외 처리하는 편이 좋다.
        raise ValueError(
            f"모델 출력에서 제거할 선택지를 파싱하지 못했습니다.\n\n출력:\n{reasoning}"
        )

    # Step 4: Remove Option
    next_option_set = [idx for idx in current_option_set if idx != eliminated_idx]

    if verbose:
        eliminated_label = chr(ord('A') + eliminated_idx)
        print("=" * 60)
        print("[Module 1] Model reasoning:")
        print(reasoning)
        print(f"\n[Module 1] Eliminated option: {eliminated_label}")
        print(f"[Module 1] Remaining options: {[chr(ord('A') + i) for i in next_option_set]}")
        print("=" * 60)

    # Step 5: return
    return next_option_set, reasoning, eliminated_idx


# -------------------------------
# 8. module_2 구현
# -------------------------------
def module_2_iterative_elimination(
    model,
    tokenizer,
    question: str,
    options: List[str],
    context: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Require: Question q, Context c, Full option set O = {o1, ..., ok}
    Ensure: Predicted answer ô
    """

    # Step 1: Initialization
    current_option_set = list(range(len(options)))
    t = 0

    history = []

    # Step 2: while |Ot| > 1 do
    while len(current_option_set) > 1:
        if verbose:
            print(f"\n[Module 2] Iteration t={t}")

        # Step 3~4: module_1 호출
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

        # Step 5: t <- t + 1
        current_option_set = next_option_set
        t += 1

    # Step 7: Final Answer Selection
    final_idx = current_option_set[0]
    final_label = chr(ord('A') + final_idx)

    # Step 8: return ô
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
# 9. k-MMLU 스타일 샘플 예시 실행
# -------------------------------
def run_example():
    tokenizer, model = load_model(MODEL_NAME)

    sample = MCQSample(
        question="한국채택국제회계기준(K-IFRS)하에서 금융자산으로 분류되지 않는 것은?",
        options=["대여금", "재고자산", "매출채권", "만기보유금융자산"],
        context=None,
    )

    result = module_2_iterative_elimination(
        model=model,
        tokenizer=tokenizer,
        question=sample.question,
        options=sample.options,
        context=sample.context,
        verbose=True
    )

    print("\n[Result Dict]")
    print(result)


if __name__ == "__main__":
    run_example()