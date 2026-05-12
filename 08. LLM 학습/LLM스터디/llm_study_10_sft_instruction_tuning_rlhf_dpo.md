# LLM 공부 10편: SFT, Instruction Tuning, RLHF, DPO

## 0. 이번 편의 목표

9편에서는 LLM의 기본 학습 방식인 **Pretraining**과 **Next Token Prediction**을 다뤘다.

이번 편에서는 Pretraining 이후에 모델을 어떻게 “사용자 지시를 잘 따르는 모델”로 바꾸는지 살펴본다.

핵심 흐름은 다음과 같다.

```text
Pretraining
→ Supervised Fine-Tuning
→ Instruction Tuning
→ Preference Alignment
→ RLHF / DPO
→ Chat / Instruct Model
```

이번 편의 핵심 질문은 이것이다.

```text
Base model은 왜 그대로 쓰기 어렵나?
SFT는 무엇을 바꾸는가?
Instruction tuning은 왜 필요한가?
RLHF는 왜 등장했나?
DPO는 RLHF와 무엇이 다른가?
Alignment는 정확히 무엇을 맞춘다는 뜻인가?
```

---

## 1. Pretraining만 끝난 모델은 어떤 상태일까?

Pretraining이 끝난 모델을 보통 **Base model**이라고 부른다.

Base model은 대규모 텍스트를 보고 다음 토큰을 예측하도록 학습된 모델이다.

즉, 기본 목적은 이것이다.

```text
주어진 문맥 뒤에 올 가능성이 높은 텍스트를 이어 쓰기
```

그래서 Base model은 언어 능력, 문법, 일반 지식, 코드 패턴 등을 어느 정도 갖고 있다.

하지만 우리가 기대하는 assistant처럼 동작한다고 보기는 어렵다.

예를 들어 사용자가 이렇게 입력했다고 하자.

```text
CVE-2021-44228에 대해 설명해줘.
```

Chat model이라면 다음처럼 답변할 것이다.

```text
CVE-2021-44228은 Apache Log4j에서 발견된 원격 코드 실행 취약점으로,
일반적으로 Log4Shell이라고 불립니다...
```

하지만 Base model은 다음처럼 동작할 수 있다.

```text
CVE-2021-44228에 대해 설명해줘.
CVE-2021-44228에 대한 분석 보고서는 다음과 같다...
```

또는 문서를 이어 쓰는 식으로 답할 수도 있다.

즉, Base model은 “질문에 답하는 법”을 명확히 배운 상태가 아니다.

---

## 2. Base model의 한계

Base model의 대표적인 한계는 다음과 같다.

```text
사용자 지시를 안정적으로 따르지 못한다.
질문에 직접 답하지 않고 문장을 이어 쓴다.
대화 형식을 유지하지 못할 수 있다.
답변 스타일이 들쭉날쭉하다.
불필요하게 장황하거나 엉뚱한 형식을 만들 수 있다.
안전하지 않은 요청에 그대로 응답할 수 있다.
```

Pretraining 데이터에는 질문-답변 형식도 있고, 문서도 있고, 코드도 있고, 게시글도 있다.

그래서 모델은 다양한 텍스트 패턴을 배웠지만,  
“사용자의 지시를 이해하고 유용한 답변을 제공하라”는 목적에 최적화된 것은 아니다.

이 문제를 해결하기 위해 **SFT**와 **Instruction Tuning**이 필요하다.

---

## 3. Supervised Fine-Tuning, SFT란?

**SFT**는 Supervised Fine-Tuning의 약자다.

한국어로는 지도 미세조정 정도로 볼 수 있다.

핵심은 사람이 준비한 정답 데이터로 모델을 추가 학습시키는 것이다.

예를 들어 다음과 같은 데이터가 있다고 하자.

```json
{
  "instruction": "CVE-2021-44228에 대해 설명해줘.",
  "response": "CVE-2021-44228은 Apache Log4j에서 발견된 원격 코드 실행 취약점입니다..."
}
```

모델은 instruction을 입력으로 받고, response를 생성하도록 학습된다.

즉, SFT의 목표는 다음과 같다.

```text
Base model이 사용자의 요청에 맞는 답변 형식을 배우게 만드는 것
```

간단히 말하면 SFT는 모델에게 이런 예시를 많이 보여주는 과정이다.

```text
사용자가 이렇게 물으면 → 이렇게 답하면 된다.
사용자가 코드를 고쳐달라고 하면 → 수정된 코드를 제공하면 된다.
사용자가 요약해달라고 하면 → 핵심만 정리하면 된다.
사용자가 표로 달라고 하면 → 표 형식으로 출력하면 된다.
```

---

## 4. SFT 데이터는 어떤 형태일까?

SFT 데이터는 보통 instruction-response 쌍으로 구성된다.

가장 단순한 형태는 다음과 같다.

```json
{
  "instruction": "다음 문장을 요약해줘.",
  "input": "LLM은 대규모 텍스트 데이터를 학습하여...",
  "output": "LLM은 대규모 텍스트로 학습된 언어 모델이다."
}
```

대화형 모델에서는 messages 형태를 많이 사용한다.

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Python에서 리스트 길이를 구하는 방법을 알려줘."
    },
    {
      "role": "assistant",
      "content": "Python에서는 len() 함수를 사용하면 됩니다. 예: len(my_list)"
    }
  ]
}
```

멀티턴 대화는 다음과 같이 구성할 수 있다.

```json
{
  "messages": [
    {
      "role": "user",
      "content": "RAG가 뭐야?"
    },
    {
      "role": "assistant",
      "content": "RAG는 검색 기반 생성 방식입니다..."
    },
    {
      "role": "user",
      "content": "보안 시스템에서는 어떻게 써?"
    },
    {
      "role": "assistant",
      "content": "보안 시스템에서는 위협 인텔리전스 문서나 로그를 검색해..."
    }
  ]
}
```

이런 데이터로 학습하면 모델은 대화 흐름과 응답 형식을 더 잘 배운다.

---

## 5. Instruction Tuning이란?

Instruction Tuning은 모델이 다양한 지시문을 잘 따르도록 학습시키는 과정이다.

SFT와 매우 밀접하게 연결되어 있다.

실무에서는 SFT와 Instruction Tuning을 거의 같은 맥락으로 말하기도 한다.

다만 개념적으로 구분하면 다음과 같다.

```text
SFT = 정답이 있는 데이터로 모델을 지도 학습시키는 방식
Instruction Tuning = 다양한 지시문을 따르도록 만드는 목적 중심의 학습
```

예를 들어 Instruction Tuning 데이터에는 이런 요청들이 포함될 수 있다.

```text
요약해줘
번역해줘
표로 정리해줘
코드를 고쳐줘
원인을 분석해줘
비교해줘
단계별로 설명해줘
JSON 형식으로 출력해줘
보안 관점에서 위험도를 평가해줘
```

즉, Instruction Tuning은 모델에게 “다양한 업무 지시를 해석하고 수행하는 법”을 가르치는 과정이다.

---

## 6. Instruction Tuning이 중요한 이유

Pretraining만 한 모델은 문장 완성 능력은 있지만, 지시 수행 능력은 부족하다.

Instruction Tuning을 하면 다음 능력이 좋아진다.

```text
사용자 의도 파악
요청 형식 준수
답변 구조화
대화 맥락 유지
작업별 출력 스타일 조정
도메인 업무 수행 방식 학습
```

예를 들어 사용자가 이렇게 요청했다고 하자.

```text
다음 incident 목록을 severity 기준으로 표로 정리해줘.
```

Instruction Tuning이 잘 된 모델은 다음을 이해해야 한다.

```text
incident 목록을 읽어야 한다.
severity를 기준으로 정렬하거나 분류해야 한다.
표 형식을 사용해야 한다.
필요한 컬럼만 뽑아야 한다.
```

이건 단순히 다음 단어를 예측하는 것보다 더 사용자 지시 중심의 행동이다.

물론 내부 학습 방식은 여전히 다음 토큰 예측 기반이지만,  
학습 데이터가 “지시 → 좋은 응답” 형식이기 때문에 행동 패턴이 바뀐다.

---

## 7. SFT만으로 충분할까?

SFT만 해도 모델은 꽤 쓸 만해진다.

하지만 SFT에는 한계가 있다.

대표적인 문제는 다음과 같다.

```text
정답 데이터 하나만 보고 학습한다.
어떤 답변이 더 좋은지 비교하는 능력은 약하다.
사람이 선호하는 답변 스타일을 세밀하게 반영하기 어렵다.
애매한 질문에서 더 안전하고 유용한 답변을 고르는 능력이 부족할 수 있다.
```

예를 들어 같은 질문에 대해 다음 두 답변이 있다고 하자.

질문:

```text
RAG와 fine-tuning의 차이를 설명해줘.
```

답변 A:

```text
RAG는 검색을 쓰는 것이고, fine-tuning은 모델을 학습시키는 것입니다.
```

답변 B:

```text
RAG는 외부 문서를 검색해 context에 넣고 답변하게 하는 방식이고,
fine-tuning은 모델 파라미터 자체를 추가 학습해 특정 도메인이나 응답 스타일에 맞추는 방식입니다.
최신 지식 반영은 RAG가 유리하고, 특정 포맷이나 말투 학습은 fine-tuning이 유리합니다.
```

둘 다 틀린 답은 아니지만, 보통 B가 더 유용하다.

SFT는 주어진 정답을 따라 하게 만드는 데 강하지만,  
여러 후보 중 어떤 답변이 더 좋은지 학습시키는 데는 한계가 있다.

그래서 **Preference Alignment**가 등장한다.

---

## 8. Preference Alignment란?

Preference Alignment는 모델의 출력이 사람의 선호와 더 잘 맞도록 조정하는 과정이다.

여기서 말하는 선호는 단순히 “예쁘게 말하기”가 아니다.

보통 다음 요소를 포함한다.

```text
정확성
유용성
명확성
간결성
안전성
지시 준수
근거 기반 답변
불확실성 표현
불필요한 환각 감소
```

예를 들어 사람 평가자가 같은 질문에 대한 두 답변을 비교한다고 하자.

```text
질문: Docker Compose에서 특정 서비스 로그를 보는 명령어는?
```

답변 A:

```text
docker compose logs -f 서비스명
```

답변 B:

```text
Docker에서는 로그를 보면 됩니다. 여러 방법이 있습니다.
```

사람은 보통 A를 더 선호할 것이다.

이런 선호 데이터를 사용해 모델을 추가로 조정하는 것이 Preference Alignment다.

---

## 9. RLHF란?

**RLHF**는 Reinforcement Learning from Human Feedback의 약자다.

한국어로는 사람 피드백 기반 강화학습이라고 볼 수 있다.

RLHF의 큰 흐름은 다음과 같다.

```text
1. SFT 모델을 준비한다.
2. 하나의 질문에 대해 여러 답변 후보를 생성한다.
3. 사람이 답변 후보의 선호도를 평가한다.
4. 선호 데이터로 Reward Model을 학습한다.
5. Reward Model이 높은 점수를 주는 방향으로 LLM을 강화학습시킨다.
```

조금 더 풀면 다음과 같다.

```text
질문: "RAG와 fine-tuning의 차이를 설명해줘."

답변 A: 짧고 부정확한 답변
답변 B: 구조적이고 정확한 답변
답변 C: 장황하고 일부 틀린 답변

사람 평가자: B > A > C
```

이런 비교 데이터를 모아 Reward Model을 만든다.

Reward Model은 답변을 보고 점수를 매긴다.

이후 LLM은 Reward Model이 높은 점수를 주는 답변을 생성하도록 추가 학습된다.

---

## 10. RLHF를 직관적으로 이해하기

RLHF를 아주 단순하게 보면 이렇다.

```text
SFT:
정답 예시를 따라 하게 만든다.

RLHF:
여러 답변 중 사람이 더 좋아하는 답변을 고르게 만든다.
```

비유하면 다음과 같다.

```text
Pretraining = 책을 엄청 많이 읽은 상태
SFT = 선생님이 문제와 모범답안을 보여준 상태
RLHF = 여러 답안 중 어떤 답안이 더 좋은지 피드백을 받은 상태
```

즉, RLHF는 모델을 단순한 문장 생성기에서 더 유용한 assistant로 다듬는 데 중요한 역할을 한다.

---

## 11. RLHF의 장점

RLHF의 장점은 다음과 같다.

```text
사람이 선호하는 답변 스타일을 반영할 수 있다.
답변의 유용성과 안전성을 높일 수 있다.
애매한 질문에서 더 적절한 답변을 선택하게 만들 수 있다.
지시 준수 능력을 개선할 수 있다.
단순 정답 데이터로 표현하기 어려운 품질 기준을 반영할 수 있다.
```

예를 들어 “좋은 설명”이라는 것은 단순한 정답 하나로 정의하기 어렵다.

좋은 설명에는 다음 요소가 함께 들어간다.

```text
정확해야 한다.
사용자 수준에 맞아야 한다.
너무 장황하지 않아야 한다.
중요한 예외를 알려줘야 한다.
실무에서 바로 쓸 수 있어야 한다.
```

RLHF는 이런 복합적인 선호를 학습시키는 데 쓰인다.

---

## 12. RLHF의 한계

RLHF도 완벽하지 않다.

대표적인 한계는 다음과 같다.

```text
구현이 복잡하다.
Reward Model을 따로 학습해야 한다.
강화학습 과정이 불안정할 수 있다.
비용이 크다.
Reward hacking 문제가 생길 수 있다.
사람 평가자의 편향이 반영될 수 있다.
```

Reward hacking은 모델이 실제로 좋은 답변을 하는 것이 아니라,  
Reward Model이 좋아할 만한 패턴만 악용하는 현상을 말한다.

예를 들어 Reward Model이 긴 답변에 높은 점수를 주는 경향이 있으면,  
모델이 불필요하게 장황한 답변을 만들 수 있다.

또한 사람의 선호 데이터 자체도 완벽하지 않다.

평가자마다 기준이 다를 수 있고, 도메인 전문성이 부족하면 잘못된 선호가 반영될 수도 있다.

---

## 13. DPO란?

**DPO**는 Direct Preference Optimization의 약자다.

RLHF보다 비교적 단순한 방식으로 preference alignment를 수행하기 위해 등장한 방법이다.

DPO의 핵심 아이디어는 다음과 같다.

```text
Reward Model을 따로 만들고 강화학습을 돌리지 않아도,
선호 데이터만으로 모델을 직접 최적화할 수 있다.
```

DPO 데이터는 보통 다음 형태다.

```json
{
  "prompt": "RAG와 fine-tuning의 차이를 설명해줘.",
  "chosen": "RAG는 외부 문서를 검색해 context에 넣는 방식이고...",
  "rejected": "RAG는 그냥 검색이고 fine-tuning은 학습입니다."
}
```

여기서 chosen은 사람이 더 선호한 답변이고, rejected는 덜 선호한 답변이다.

DPO는 모델이 chosen 답변의 확률을 높이고, rejected 답변의 확률을 낮추도록 학습한다.

---

## 14. DPO를 직관적으로 이해하기

DPO는 이렇게 이해하면 된다.

```text
이 질문에는 이 답변이 더 좋고,
저 답변은 덜 좋다.
그러니 좋은 답변을 생성할 확률을 높이고,
나쁜 답변을 생성할 확률은 낮춰라.
```

RLHF와 비교하면 다음과 같다.

```text
RLHF:
선호 데이터 → Reward Model 학습 → 강화학습으로 LLM 조정

DPO:
선호 데이터 → LLM을 직접 조정
```

DPO는 구조가 단순하고 안정적인 편이라 실무와 오픈소스 모델 튜닝에서 자주 언급된다.

---

## 15. RLHF와 DPO 비교

| 구분 | RLHF | DPO |
|---|---|---|
| 전체 구조 | 복잡함 | 상대적으로 단순함 |
| Reward Model | 필요 | 별도 Reward Model 불필요 |
| 강화학습 | 사용 | 사용하지 않음 |
| 데이터 형태 | 선호 비교 데이터 | chosen/rejected 선호 데이터 |
| 구현 난이도 | 높음 | 상대적으로 낮음 |
| 안정성 | 튜닝이 어려울 수 있음 | 비교적 안정적 |
| 목적 | 사람 선호 반영 | 사람 선호 반영 |

단순하게 말하면 다음과 같다.

```text
RLHF는 더 복잡하지만 강력한 전통적 alignment 방식이다.
DPO는 선호 데이터를 더 직접적으로 활용하는 단순한 alignment 방식이다.
```

---

## 16. Chat Model은 어떻게 만들어지는가?

Chat model은 보통 다음 단계를 거쳐 만들어진다.

```text
1. Pretraining
   대규모 텍스트로 기본 언어 능력 학습

2. SFT / Instruction Tuning
   사용자 지시와 좋은 답변 예시를 학습

3. Preference Alignment
   사람이 선호하는 답변을 더 잘 생성하도록 조정

4. Safety Tuning
   위험하거나 부적절한 요청에 안전하게 대응하도록 조정

5. Evaluation
   벤치마크, 사람 평가, 도메인 테스트로 품질 확인
```

이 과정을 거치면 모델은 다음 능력을 갖게 된다.

```text
사용자 질문에 직접 답하기
대화 맥락 유지하기
요청한 형식 따르기
불확실한 정보에 대해 조심스럽게 답하기
위험한 요청 거절하기
코드 작성, 요약, 번역, 분석 등 다양한 작업 수행하기
```

---

## 17. Base Model, Instruct Model, Chat Model 차이

| 구분 | 설명 | 주 사용처 |
|---|---|---|
| Base Model | Pretraining 중심의 기본 모델 | 추가 학습, 연구, completion |
| Instruct Model | 지시문을 잘 따르도록 튜닝된 모델 | 일반 질의응답, 작업 수행 |
| Chat Model | 대화 형식에 최적화된 모델 | 챗봇, assistant, multi-turn 대화 |

예를 들어 같은 모델 계열이라도 다음처럼 나뉠 수 있다.

```text
Llama base
Llama instruct
Qwen base
Qwen instruct
Mistral base
Mistral instruct
```

일반적으로 애플리케이션을 만들 때는 base model보다 instruct/chat model을 사용하는 것이 좋다.

특히 RAG, Agent, Tool Calling 시스템에서는 지시 준수 능력이 중요하기 때문에 instruct/chat model이 훨씬 유리하다.

---

## 18. Fine-tuning과 RAG는 다른 문제를 해결한다

많이 헷갈리는 부분이 있다.

```text
“도메인 지식이 부족하면 fine-tuning을 해야 하나?”
```

항상 그렇지는 않다.

Fine-tuning과 RAG는 해결하는 문제가 다르다.

| 구분 | Fine-tuning | RAG |
|---|---|---|
| 목적 | 모델의 행동, 스타일, 특정 패턴 학습 | 외부 지식 검색 및 근거 제공 |
| 지식 업데이트 | 재학습 필요 | 문서 업데이트로 가능 |
| 최신 정보 반영 | 불리함 | 유리함 |
| 응답 형식 학습 | 유리함 | 제한적 |
| 사내 문서 기반 답변 | 단독으로는 위험 | 적합 |
| 비용 | 학습 비용 발생 | 검색 시스템 구축 비용 발생 |

예를 들어 보안 업무에서 다음과 같이 나눌 수 있다.

```text
RAG가 적합한 경우:
- 최신 CVE 문서를 참조해야 한다.
- 사내 보안 정책을 근거로 답해야 한다.
- 침해사고 보고서를 기반으로 요약해야 한다.
- TI DB의 최신 정보를 참조해야 한다.

Fine-tuning이 적합한 경우:
- 항상 특정 보고서 형식으로 답해야 한다.
- 보안 분석가처럼 판단하는 스타일을 학습시키고 싶다.
- Text-to-SQL 쿼리 생성 패턴을 도메인에 맞추고 싶다.
- 특정 조직의 응답 톤과 업무 절차를 익히게 하고 싶다.
```

가장 현실적인 시스템은 둘 중 하나만 쓰는 것이 아니라, 보통 함께 쓴다.

```text
RAG로 근거를 가져오고,
Instruct/Fine-tuned model로 원하는 형식의 답변을 생성한다.
```

---

## 19. 네 프로젝트에 연결해서 이해하기

네가 구축하는 MDR multi-agent 시스템에 연결하면 이렇게 볼 수 있다.

```text
Pretraining:
모델이 기본적인 언어, 보안 용어, 코드, 분석 표현을 배운 상태

Instruction Tuning:
사용자 질문을 이해하고, “incident 목록 보여줘”, “분석해줘”, “SQL로 조회해줘” 같은 지시를 따르는 능력

RAG:
보안 문서, 정책, 취약점 설명, TI 보고서 등 외부 문서를 검색해서 근거 제공

Tool Calling:
Cortex XDR MCP, PostgreSQL, API 등을 실제로 호출

Preference Alignment:
더 유용하고, 안전하고, 명확한 답변을 선호하도록 조정된 상태
```

예를 들어 사용자가 이렇게 물었다고 하자.

```text
최근 7일 high severity incident를 정리하고 관련 TI 정보까지 같이 분석해줘.
```

좋은 시스템은 다음처럼 동작한다.

```text
1. 사용자 의도 파악
2. XDR MCP tool 호출
3. incident 목록 수집
4. T2S agent가 TI DB 조회
5. RAG로 관련 문서 검색
6. analysis agent가 결과 종합
7. 표와 권고사항 형태로 응답
```

여기서 모델이 instruction tuning이 잘 되어 있지 않으면 도구를 써야 하는 상황을 놓치거나, 사용자가 원하는 형식을 지키지 못할 수 있다.

---

## 20. SFT 데이터 설계 예시: 보안 도메인

보안 도메인 assistant를 만든다고 하면 SFT 데이터는 이런 식으로 설계할 수 있다.

### 예시 1: Incident 요약

```json
{
  "messages": [
    {
      "role": "user",
      "content": "다음 incident를 분석해서 요약해줘."
    },
    {
      "role": "assistant",
      "content": "요약, 영향 범위, 의심 원인, 권고 조치 순서로 정리하겠습니다..."
    }
  ]
}
```

### 예시 2: SQL 생성

```json
{
  "messages": [
    {
      "role": "user",
      "content": "ti_ioc_ip 테이블에서 최근 7일간 등록된 악성 IP를 조회하는 SQL을 만들어줘."
    },
    {
      "role": "assistant",
      "content": "다음 SQL을 사용할 수 있습니다:\n\nSELECT * FROM public.ti_ioc_ip WHERE created_at >= NOW() - INTERVAL '7 days';"
    }
  ]
}
```

### 예시 3: 보안 보고서 형식

```json
{
  "messages": [
    {
      "role": "user",
      "content": "다음 탐지 결과를 일일 보안 보고서 형식으로 정리해줘."
    },
    {
      "role": "assistant",
      "content": "## 1. 요약\n...\n## 2. 주요 탐지\n...\n## 3. 조치 권고\n..."
    }
  ]
}
```

이런 데이터는 모델이 특정 업무 스타일을 배우는 데 도움이 된다.

---

## 21. Preference 데이터 설계 예시

DPO나 RLHF용 선호 데이터는 chosen/rejected 형태로 만들 수 있다.

예를 들어 다음과 같다.

```json
{
  "prompt": "최근 high severity incident를 요약해줘.",
  "chosen": "최근 7일 기준 high severity incident는 총 3건입니다. 주요 영향 자산, 사용자, 권고 조치는 다음과 같습니다...",
  "rejected": "High severity incident가 있습니다. 확인이 필요합니다."
}
```

chosen이 좋은 이유는 다음과 같다.

```text
구체적이다.
개수가 명확하다.
분석 기준이 있다.
권고 조치가 포함되어 있다.
```

rejected가 나쁜 이유는 다음과 같다.

```text
너무 모호하다.
사용자가 바로 행동할 수 없다.
근거와 세부 정보가 없다.
```

이런 선호 데이터를 많이 만들면 모델은 “좋은 보안 분석 답변”의 기준을 더 잘 배운다.

---

## 22. Fine-tuning할 때 주의할 점

Fine-tuning은 강력하지만, 무작정 하면 안 된다.

주의할 점은 다음과 같다.

```text
데이터 품질이 낮으면 모델도 나빠진다.
잘못된 답변 형식을 학습할 수 있다.
과도하게 특정 패턴에만 맞춰질 수 있다.
최신 지식 주입 용도로는 비효율적일 수 있다.
민감정보가 학습 데이터에 들어가면 위험하다.
평가 데이터 없이 튜닝하면 개선 여부를 알 수 없다.
```

특히 실무에서는 다음 원칙이 중요하다.

```text
문서 지식은 RAG로 처리한다.
반복되는 응답 형식과 업무 절차는 fine-tuning 후보로 본다.
실시간 데이터는 tool calling으로 가져온다.
모델 튜닝 전후 평가셋을 반드시 만든다.
```

Fine-tuning은 만능 해결책이 아니다.

잘 쓰면 강력하지만, 잘못 쓰면 비용만 쓰고 품질은 떨어질 수 있다.

---

## 23. Evaluation이 반드시 필요한 이유

SFT나 DPO를 했다고 해서 모델이 반드시 좋아졌다고 볼 수는 없다.

반드시 평가가 필요하다.

평가 항목은 다음과 같이 나눌 수 있다.

```text
정확성
지시 준수
응답 형식 준수
도메인 용어 사용
근거 기반 답변
환각 여부
안전성
일관성
도구 호출 판단 정확도
SQL 생성 정확도
```

보안 도메인에서는 특히 다음 평가가 중요하다.

```text
실제 존재하지 않는 CVE를 만들어내지 않는가?
IOC와 정상 IP를 혼동하지 않는가?
SQL 테이블명을 정확히 사용하는가?
권고 조치가 현실적인가?
위험도를 과장하거나 축소하지 않는가?
로그 근거 없이 단정하지 않는가?
```

LLM 시스템을 만들 때는 모델 자체보다 평가 체계가 더 중요해지는 경우가 많다.

---

## 24. 실습 아이디어

이번 편을 이해했다면 다음 실습을 해보면 좋다.

### 실습 1: Base model과 Instruct model 비교

같은 계열의 base model과 instruct model에 같은 질문을 넣어본다.

예시 질문:

```text
RAG와 fine-tuning의 차이를 표로 설명해줘.
```

비교할 점:

```text
지시를 잘 따르는가?
표 형식을 지키는가?
답변이 직접적인가?
불필요한 문장 이어쓰기가 있는가?
```

### 실습 2: 작은 SFT 데이터 만들어보기

10개 정도의 instruction-response 데이터를 만들어본다.

예시:

```text
SQL 오류 원인 분석
보안 로그 요약
CVE 설명
Docker 명령어 설명
RAG chunking 전략 설명
```

그리고 이 데이터가 모델에게 어떤 행동을 가르치는지 생각해본다.

### 실습 3: Chosen/Rejected 데이터 만들기

같은 질문에 대해 좋은 답변과 나쁜 답변을 직접 만들어본다.

예시:

```json
{
  "prompt": "PostgreSQL에서 cross-database references 에러가 나는 이유는?",
  "chosen": "PostgreSQL은 MySQL처럼 USE database로 DB를 전환하지 않으며...",
  "rejected": "DB 이름을 바꾸면 됩니다."
}
```

이렇게 만들면 preference alignment 데이터의 감을 잡을 수 있다.

---

## 25. 자주 헷갈리는 포인트

### 25.1 SFT와 Instruction Tuning은 다른가?

엄밀히는 다르지만 실무에서는 많이 겹쳐서 쓰인다.

```text
SFT = 지도학습 방식
Instruction Tuning = 지시를 따르게 만드는 목적
```

Instruction Tuning은 보통 SFT 방식으로 수행된다.

### 25.2 RLHF를 꼭 해야 하나?

항상 필요한 것은 아니다.

작은 도메인 모델이나 내부 업무용 모델은 SFT와 RAG만으로도 충분할 수 있다.

하지만 대규모 범용 assistant에서는 사람 선호와 안전성을 반영하기 위해 RLHF나 DPO 같은 alignment가 중요해진다.

### 25.3 DPO가 RLHF보다 무조건 좋은가?

무조건 그렇지는 않다.

DPO는 단순하고 실용적이지만, 데이터 품질과 목적에 따라 결과가 달라진다.

RLHF, DPO 모두 핵심은 좋은 선호 데이터다.

### 25.4 Fine-tuning으로 최신 지식을 넣으면 되지 않나?

가능은 하지만 보통 비효율적이다.

최신 정보나 자주 바뀌는 사내 데이터는 RAG나 tool calling이 더 적합하다.

Fine-tuning은 지식 주입보다 행동 패턴, 응답 형식, 도메인 작업 스타일 학습에 더 적합하다.

### 25.5 Alignment는 검열인가?

Alignment에는 안전성 조정도 포함되지만, 그것만 의미하지는 않는다.

더 넓게는 모델의 행동을 사람이 원하는 방향에 맞추는 과정이다.

```text
더 유용하게
더 정확하게
더 안전하게
더 명확하게
더 지시를 잘 따르게
```

이 모든 것이 alignment에 포함된다.

---

## 26. 이번 편 핵심 요약

이번 편의 핵심은 다음과 같다.

```text
Pretraining은 Base model을 만든다.
Base model은 텍스트를 이어 쓰는 능력은 있지만 assistant처럼 동작하기 어렵다.
SFT는 정답 예시를 통해 사용자 지시와 좋은 응답 형식을 학습시킨다.
Instruction Tuning은 다양한 지시를 잘 따르게 만드는 과정이다.
SFT만으로는 사람의 세밀한 선호를 반영하기 어렵다.
RLHF는 사람의 선호 데이터를 Reward Model과 강화학습으로 반영한다.
DPO는 chosen/rejected 선호 데이터를 사용해 모델을 직접 최적화한다.
Fine-tuning은 최신 지식 주입보다 행동 패턴과 응답 스타일 학습에 더 적합하다.
RAG, Tool Calling, Fine-tuning은 서로 대체재가 아니라 보완재다.
```

한 문장으로 정리하면 이렇다.

```text
Pretraining이 모델의 기본 지능을 만든다면, SFT와 Alignment는 그 지능을 사용자가 원하는 방식으로 꺼내 쓰게 만드는 과정이다.
```

---

## 27. 다음 편 예고

다음 11편에서는 **Decoding과 Sampling**을 다루는 것이 좋다.

다음 질문들을 중심으로 보면 된다.

```text
LLM은 다음 토큰을 어떻게 선택하는가?
Greedy decoding이란 무엇인가?
Temperature는 무엇을 바꾸는가?
Top-k와 top-p는 무엇이 다른가?
왜 같은 질문에도 매번 다른 답변이 나오는가?
실무 챗봇에서는 temperature를 어떻게 잡는 것이 좋은가?
```

이 부분을 이해하면 LLM 답변이 왜 달라지는지, 그리고 서비스에서 생성 옵션을 어떻게 조정해야 하는지 감이 잡히기 시작한다.
