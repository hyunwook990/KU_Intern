# MAS-GPT practice
- 출처: ttps://arxiv.org/abs/2503.03686

### 2026.03.09
- 논문에서 나온 내용중 구현에 참고할 내용들을 확인
1. Following this framework, we first re-implement several existing MAS methods (e.g., Multi-Agent Debate (Duet al., 2024), Self-Consistency (Wang et al., 2024b), Self-Refine (Madaan et al., 2024)) to align with our unified code representation.
2. To further expand the diversity of MAS candidates, we also manually design some MAS systems, resulting in a base MAS pool comprising over 40 unique MAS designs
3. Importantly, these 40+ MAS do not directly correspond to the exact number of MAS in the training dataset; rather, they serve as foundations that evolve during the query-MAS pair refinement process.
- github `template.py`에 MAS pyhton code 생성 System prompt로 보이는 것을 발견
- 데이터 생성: Llama-3-70B-Instruct 사용
- MAS-GPT backbone: Qwen2.5-Coder-32B-Instruct 사용 (to leveraging instruction-following, coding capabilities)
### 2026.03.10
- MAS-GPT github 주소에 template.py
### 2026.03.12
- 