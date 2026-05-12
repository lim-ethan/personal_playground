# LLM 공부 7편: GPT와 Decoder-only Transformer

## 0. 이번 편의 목표

이번 편에서는 GPT 계열 모델의 기본 구조인 **Decoder-only Transformer**를 이해한다.

앞에서 우리는 다음 흐름을 봤다.

```text
RNN / LSTM
→ Seq2Seq
→ Attention
→ Transformer
→ Self-Attention / Multi-Head Attention
```

이제 여기서 한 단계 더 나아가야 한다.

GPT는 Transformer를 그대로 쓰는 모델이 아니다.  
정확히 말하면 Transformer 구조 중에서 **Decoder 쪽 구조를 중심으로 사용한 생성 모델**이다.

이번 편의 핵심 질문은 다음과 같다.

```text
GPT는 왜 Decoder-only 구조를 사용할까?
GPT는 문장을 어떻게 생성할까?
BERT와 GPT는 무엇이 다를까?
Causal Masking은 왜 필요할까?
Next Token Prediction은 무엇인가?
```

---

## 1. GPT란 무엇인가?

GPT는 **Generative Pre-trained Transformer**의 약자다.

풀어보면 다음과 같다.

```text
Generative  : 텍스트를 생성할 수 있다.
Pre-trained : 대규모 텍스트로 미리 학습되어 있다.
Transformer : Transformer 구조를 기반으로 한다.
```

즉 GPT는 대규모 텍스트를 바탕으로 미리 학습된 Transformer 기반 생성 모델이다.

중요한 점은 GPT가 단순히 문장을 이해하는 모델이 아니라는 것이다.

GPT의 주된 목적은 다음과 같다.

```text
주어진 문맥을 보고 다음 토큰을 예측한다.
```

예를 들어 입력이 다음과 같다고 하자.

```text
오늘 날씨가 정말
```

모델은 다음에 올 가능성이 높은 토큰을 예측한다.

```text
좋다
춥다
맑다
덥다
```

그리고 하나의 토큰을 선택한 뒤, 다시 그 토큰까지 포함해서 다음 토큰을 예측한다.

```text
오늘 날씨가 정말 좋다
→ 다음 토큰 예측

오늘 날씨가 정말 좋다.
→ 다음 토큰 예측
```

이 과정을 반복하면 문장이 생성된다.

---

## 2. GPT의 가장 중요한 특징

GPT의 핵심 특징은 다음 세 가지다.

```text
1. Decoder-only Transformer 구조를 사용한다.
2. 왼쪽에서 오른쪽 방향으로 텍스트를 생성한다.
3. Next Token Prediction 방식으로 학습한다.
```

이 세 가지를 이해하면 GPT 계열 LLM의 기본 동작을 설명할 수 있다.

---

## 3. Transformer는 원래 Encoder-Decoder 구조였다

원래 Transformer 논문에서 제안된 구조는 Encoder와 Decoder를 모두 가진 구조였다.

```text
입력 문장 → Encoder → 의미 표현 → Decoder → 출력 문장
```

기계번역을 예로 들면 다음과 같다.

```text
입력: I love you
출력: 나는 너를 사랑한다
```

이때 Encoder는 입력 문장을 이해하고, Decoder는 출력 문장을 생성한다.

구조적으로 보면 다음과 같다.

```text
[Encoder]
입력 문장을 읽고 문맥 표현을 만든다.

[Decoder]
Encoder의 결과를 참고하면서 출력 문장을 하나씩 생성한다.
```

하지만 GPT는 이 전체 구조를 그대로 쓰지 않는다.

GPT는 Encoder 없이 **Decoder 구조만 사용한다**.

그래서 GPT를 다음과 같이 부른다.

```text
Decoder-only Transformer
```

---

## 4. Encoder-only, Decoder-only, Encoder-Decoder 비교

Transformer 계열 모델은 크게 세 가지 구조로 나눌 수 있다.

| 구조 | 대표 모델 | 주 용도 |
|---|---|---|
| Encoder-only | BERT | 문장 이해, 분류, 검색, 임베딩 |
| Decoder-only | GPT | 텍스트 생성, 대화, 코드 생성 |
| Encoder-Decoder | T5, BART | 번역, 요약, 변환 작업 |

각 구조의 차이를 조금 더 보면 다음과 같다.

## 4.1 Encoder-only

Encoder-only 모델은 입력 전체를 양방향으로 본다.

예를 들어 다음 문장이 있다고 하자.

```text
나는 오늘 [MASK]를 먹었다.
```

BERT는 `[MASK]` 앞뒤 문맥을 모두 보고 빈칸을 예측한다.

```text
앞 문맥: 나는 오늘
뒤 문맥: 를 먹었다
```

이런 구조는 문장 이해에 강하다.

예를 들면 다음 작업에 잘 맞는다.

```text
문장 분류
감성 분석
문서 검색
문장 임베딩
개체명 인식
```

하지만 자연스러운 긴 문장을 왼쪽에서 오른쪽으로 계속 생성하는 데는 GPT보다 적합하지 않다.

## 4.2 Decoder-only

Decoder-only 모델은 이전 토큰만 보고 다음 토큰을 예측한다.

예를 들어 다음과 같다.

```text
입력: 나는 오늘
예측: 점심

입력: 나는 오늘 점심
예측: 으로

입력: 나는 오늘 점심으로
예측: 김치찌개
```

이 구조는 텍스트 생성에 강하다.

그래서 GPT 계열 모델은 다음 작업에 잘 맞는다.

```text
대화형 챗봇
문서 생성
코드 생성
요약
질문 답변
Agent reasoning
Tool calling
```

## 4.3 Encoder-Decoder

Encoder-Decoder 구조는 입력과 출력이 명확히 구분되는 작업에 적합하다.

예를 들면 다음과 같다.

```text
영어 문장 → 한국어 문장
긴 문서 → 짧은 요약
질문 → 정답 문장
```

T5나 BART 같은 모델이 이 계열에 속한다.

---

## 5. GPT는 왜 Decoder-only 구조를 사용할까?

GPT의 목표는 텍스트 생성이다.

텍스트 생성은 기본적으로 다음과 같은 과정이다.

```text
지금까지 생성된 토큰을 보고 다음 토큰을 예측한다.
```

이 작업에는 Encoder가 반드시 필요하지 않다.

입력 문장 전체를 별도의 Encoder로 이해한 뒤 Decoder가 생성하는 구조보다, 하나의 Decoder stack만으로 문맥을 누적하면서 다음 토큰을 예측하는 방식이 더 단순하고 확장하기 좋다.

GPT 입장에서 모든 입력은 하나의 긴 토큰 시퀀스다.

```text
시스템 메시지 + 사용자 질문 + 이전 대화 + 도구 결과 + 현재 답변 일부
```

모델은 이 전체 시퀀스를 보고 다음 토큰을 예측한다.

즉, GPT에게는 질문과 답변의 구분도 결국 토큰 패턴이다.

```text
User: 최근 7일 incident 목록 보여줘
Assistant: 최근 7일 동안 확인된 incident는 ...
```

이런 대화 형식도 학습 데이터 안에서 반복적으로 본 패턴이기 때문에 이어서 답변을 생성할 수 있다.

---

## 6. GPT의 입력과 출력

GPT의 입력은 텍스트가 아니라 토큰 ID다.

예를 들어 사용자가 다음과 같이 입력한다고 하자.

```text
최근 7일 incident 목록 보여줘
```

모델 내부에서는 대략 다음 과정을 거친다.

```text
텍스트
→ Tokenizer
→ Token ID sequence
→ Token Embedding
→ Transformer blocks
→ 다음 토큰 확률 분포
```

출력도 처음부터 완성된 문장이 아니다.

모델은 매 단계마다 다음 토큰 하나에 대한 확률 분포를 만든다.

```text
P(token | 이전 토큰들)
```

예를 들면 다음과 같다.

```text
입력: 최근 7일 incident 목록

다음 토큰 후보:
- 을: 0.31
- 은: 0.18
- 입니다: 0.07
- 확인: 0.05
- 보여: 0.03
```

이 중 하나를 선택하고, 다시 다음 토큰을 예측한다.

이 과정을 반복해서 최종 답변이 만들어진다.

---

## 7. Next Token Prediction

GPT 학습의 핵심은 **Next Token Prediction**이다.

말 그대로 다음 토큰을 맞히는 방식이다.

예를 들어 학습 문장이 다음과 같다고 하자.

```text
나는 오늘 점심으로 김치찌개를 먹었다.
```

모델은 학습 중에 다음과 같은 문제를 계속 푼다.

```text
나는 → 오늘
나는 오늘 → 점심으로
나는 오늘 점심으로 → 김치찌개를
나는 오늘 점심으로 김치찌개를 → 먹었다
```

정리하면 다음과 같다.

```text
입력 토큰들: x1, x2, x3, ... x(n-1)
정답 토큰: xn
```

모델은 매 위치에서 다음 토큰을 예측하고, 정답과 비교해 loss를 계산한다.

이 방식은 단순해 보이지만 매우 강력하다.

왜냐하면 인터넷 텍스트, 문서, 코드, 대화, 논문, 설명문 등 거의 모든 텍스트를 이 방식으로 학습 데이터로 사용할 수 있기 때문이다.

---

## 8. Causal Masking

GPT는 다음 토큰을 예측할 때 미래 토큰을 보면 안 된다.

예를 들어 다음 문장을 학습한다고 하자.

```text
나는 오늘 점심으로 김치찌개를 먹었다.
```

모델이 `점심으로`를 예측할 때 이미 뒤에 있는 `김치찌개를 먹었다`를 볼 수 있다면 반칙이다.

그래서 GPT는 Attention을 계산할 때 미래 토큰을 보지 못하게 막는다.

이것을 **Causal Masking**이라고 한다.

쉽게 말하면 다음 규칙이다.

```text
현재 위치의 토큰은 자기 자신과 이전 토큰만 볼 수 있다.
미래 위치의 토큰은 볼 수 없다.
```

예를 들어 토큰이 다음과 같다고 하자.

```text
[나는] [오늘] [점심으로] [김치찌개를] [먹었다]
```

각 토큰이 볼 수 있는 범위는 다음과 같다.

| 위치 | 토큰 | 볼 수 있는 토큰 |
|---|---|---|
| 1 | 나는 | 나는 |
| 2 | 오늘 | 나는, 오늘 |
| 3 | 점심으로 | 나는, 오늘, 점심으로 |
| 4 | 김치찌개를 | 나는, 오늘, 점심으로, 김치찌개를 |
| 5 | 먹었다 | 나는, 오늘, 점심으로, 김치찌개를, 먹었다 |

이 구조 때문에 GPT는 왼쪽에서 오른쪽으로 자연스럽게 문장을 생성할 수 있다.

---

## 9. Self-Attention과 Causal Self-Attention의 차이

일반 Self-Attention은 모든 토큰이 서로를 볼 수 있다.

```text
토큰 A ↔ 토큰 B ↔ 토큰 C ↔ 토큰 D
```

하지만 GPT의 Causal Self-Attention은 미래 방향을 막는다.

```text
토큰 A → A만 봄
토큰 B → A, B만 봄
토큰 C → A, B, C만 봄
토큰 D → A, B, C, D만 봄
```

이 차이가 중요하다.

BERT 같은 Encoder-only 모델은 양방향 문맥을 본다.

```text
앞 문맥 + 뒤 문맥 모두 사용
```

GPT 같은 Decoder-only 모델은 단방향 문맥을 본다.

```text
이전 문맥만 사용
```

그래서 BERT는 문장 이해에 강하고, GPT는 생성에 강하다.

---

## 10. GPT Block의 기본 구조

GPT는 여러 개의 Transformer block을 쌓은 구조다.

하나의 GPT block은 보통 다음 구성요소를 가진다.

```text
1. Layer Normalization
2. Causal Self-Attention
3. Residual Connection
4. Layer Normalization
5. Feed Forward Network
6. Residual Connection
```

단순화하면 다음과 같다.

```text
입력
→ LayerNorm
→ Causal Self-Attention
→ Residual Add
→ LayerNorm
→ Feed Forward Network
→ Residual Add
→ 출력
```

이 block을 수십 개, 수백 개 쌓으면 GPT 계열 모델이 된다.

작은 모델은 layer 수가 적고, 큰 모델은 layer 수와 hidden dimension, attention head 수가 많다.

---

## 11. GPT의 전체 흐름

GPT의 전체 동작을 하나로 정리하면 다음과 같다.

```text
1. 사용자의 입력 텍스트를 받는다.
2. Tokenizer가 텍스트를 토큰 ID로 바꾼다.
3. 각 토큰 ID를 embedding vector로 변환한다.
4. 위치 정보를 추가한다.
5. 여러 개의 Decoder Transformer block을 통과한다.
6. 마지막 hidden state를 vocabulary 크기의 확률 분포로 변환한다.
7. 다음 토큰을 선택한다.
8. 선택된 토큰을 입력 뒤에 붙인다.
9. 다시 1~8 과정을 반복한다.
```

이것을 더 짧게 표현하면 다음과 같다.

```text
토큰화 → 임베딩 → Decoder blocks → 다음 토큰 확률 → 샘플링 → 반복
```

---

## 12. Vocabulary와 Logits

GPT는 다음 토큰을 예측할 때 모든 단어를 직접 고르는 것이 아니다.

모델은 vocabulary에 있는 모든 토큰에 대해 점수를 계산한다.

이 점수를 **logits**라고 한다.

예를 들어 vocabulary 크기가 100,000개라면, 모델은 다음 토큰 후보 100,000개에 대한 점수를 만든다.

```text
logits = [2.1, -0.3, 5.7, 1.2, ...]
```

이 logits에 softmax를 적용하면 확률이 된다.

```text
probabilities = softmax(logits)
```

그 다음 decoding 전략에 따라 다음 토큰을 고른다.

대표적인 전략은 다음과 같다.

```text
Greedy decoding
Temperature sampling
Top-k sampling
Top-p sampling
Beam search
```

이 내용은 다음 편들에서 더 자세히 볼 수 있다.

---

## 13. GPT가 문장을 생성하는 방식

GPT가 답변을 생성하는 과정을 예시로 보자.

입력:

```text
LLM은 무엇인가요?
```

모델이 생성하는 과정은 실제로는 다음과 비슷하다.

```text
LLM은
LLM은 대규모
LLM은 대규모 언어
LLM은 대규모 언어 모델로,
LLM은 대규모 언어 모델로, 많은
LLM은 대규모 언어 모델로, 많은 텍스트
...
```

사람 눈에는 한 문장이 한 번에 나오는 것처럼 보이지만, 내부적으로는 토큰을 하나씩 생성한다.

Streaming 응답도 이 원리와 연결된다.

서버가 모델에서 생성된 토큰 또는 텍스트 조각을 받을 때마다 UI로 전달하면, 사용자는 답변이 실시간으로 생성되는 것처럼 보게 된다.

---

## 14. GPT와 ChatGPT의 차이

GPT와 ChatGPT는 같은 말처럼 쓰이지만 엄밀히는 다르다.

```text
GPT
: 모델 구조 또는 모델 계열을 의미한다.

ChatGPT
: GPT 계열 모델을 대화형 서비스로 만든 제품 또는 인터페이스를 의미한다.
```

즉 GPT는 엔진에 가깝고, ChatGPT는 그 엔진을 사용한 대화형 서비스에 가깝다.

또한 실제 ChatGPT 같은 서비스는 단순 GPT 모델만 있는 것이 아니다.

보통 다음 요소들이 함께 붙는다.

```text
System prompt
Instruction tuning
Safety policy
Tool calling
Memory
RAG
Function calling
Streaming API
User interface
```

네가 만들고 있는 Agent 시스템도 이 관점에서 보면 LLM을 중심으로 여러 기능을 조합한 애플리케이션이라고 볼 수 있다.

---

## 15. Base Model과 Instruct Model

GPT 계열 모델을 공부할 때 중요한 구분이 있다.

```text
Base Model
Instruct Model
Chat Model
```

## 15.1 Base Model

Base Model은 기본적으로 다음 토큰 예측만 학습한 모델이다.

예를 들어 사용자가 이렇게 입력한다고 하자.

```text
질문: 서울의 수도는?
답변:
```

Base Model은 질문에 친절하게 답한다기보다, 학습 데이터에서 그럴듯하게 이어질 텍스트를 생성한다.

즉, 지시를 따른다기보다는 텍스트 패턴을 이어간다.

## 15.2 Instruct Model

Instruct Model은 사용자의 지시를 따르도록 추가 학습된 모델이다.

예를 들어 다음과 같은 지시를 더 잘 따른다.

```text
다음 내용을 요약해줘.
SQL을 작성해줘.
보안 이벤트를 분석해줘.
Python 코드로 만들어줘.
```

일반적으로 우리가 챗봇, Agent, 업무 자동화에 사용하는 모델은 Base Model보다 Instruct 또는 Chat Model이 더 적합하다.

## 15.3 Chat Model

Chat Model은 대화 형식에 맞게 튜닝된 모델이다.

보통 입력이 다음과 같은 역할 구조를 가진다.

```text
system
user
assistant
tool
```

이런 구조는 멀티턴 대화, 도구 호출, 시스템 지침 반영에 유리하다.

---

## 16. GPT와 BERT 비교

GPT와 BERT의 차이는 자주 헷갈린다.

가장 중요한 차이는 다음과 같다.

| 항목 | GPT | BERT |
|---|---|---|
| 구조 | Decoder-only | Encoder-only |
| 방향성 | 단방향 | 양방향 |
| 학습 방식 | Next Token Prediction | Masked Language Modeling |
| 강점 | 생성 | 이해 |
| 대표 용도 | 대화, 글쓰기, 코드 생성 | 분류, 검색, 임베딩 |

예를 들어 RAG 시스템에서 문서 임베딩을 만들 때는 BERT 계열 embedding model이 많이 쓰인다.

반면 검색된 문서를 바탕으로 최종 답변을 생성할 때는 GPT 계열 생성 모델이 많이 쓰인다.

즉 RAG에서는 두 계열이 함께 쓰이기도 한다.

```text
Embedding model: 문서와 질문을 벡터화
LLM: 검색 결과를 바탕으로 답변 생성
```

---

## 17. GPT가 지식을 저장하는 방식

GPT는 데이터베이스처럼 지식을 저장하지 않는다.

예를 들어 모델 내부에 다음과 같은 테이블이 있는 것이 아니다.

```text
국가 | 수도
한국 | 서울
일본 | 도쿄
프랑스 | 파리
```

대신 대규모 학습 과정에서 언어 패턴과 사실 관계가 파라미터에 분산되어 저장된다.

그래서 GPT의 지식은 다음과 같은 특징을 가진다.

```text
명시적이지 않다.
정확한 출처를 기본적으로 알지 못한다.
학습 시점 이후의 정보는 모를 수 있다.
비슷한 패턴을 바탕으로 그럴듯한 답을 만들 수 있다.
```

이 때문에 LLM은 hallucination을 만들 수 있다.

RAG, Tool Calling, DB 조회, 웹 검색 같은 구조가 필요한 이유도 여기에 있다.

---

## 18. GPT와 RAG의 연결

GPT는 자체 파라미터에 저장된 지식만으로 답할 수도 있지만, 실무 시스템에서는 외부 지식을 붙이는 경우가 많다.

RAG는 다음 구조다.

```text
사용자 질문
→ 관련 문서 검색
→ 검색 결과를 프롬프트에 포함
→ GPT가 답변 생성
```

GPT 입장에서 검색 결과도 그냥 입력 토큰의 일부다.

예를 들어 다음과 같이 들어간다.

```text
[사용자 질문]
최근 발생한 취약점 중 위험도가 높은 항목을 알려줘.

[검색된 문서]
CVE-XXXX는 EPSS 점수가 높고 실제 공격에 사용된 이력이 있다.

[지시]
검색된 문서를 근거로 답변하라.
```

GPT는 이 전체 문맥을 보고 다음 토큰을 생성한다.

즉 RAG는 GPT의 구조를 바꾸는 것이 아니라, GPT에게 더 좋은 문맥을 제공하는 방식이다.

---

## 19. GPT와 Agent의 연결

Agent 구조에서도 GPT는 핵심 역할을 한다.

Agent에서 GPT는 보통 다음 판단을 수행한다.

```text
사용자 요청 이해
필요한 도구 선택
도구 호출 결과 해석
다음 행동 결정
최종 답변 생성
```

예를 들어 네 MDR 플랫폼 관점에서 보면 다음과 같다.

```text
사용자: 최근 7일 high severity incident 목록 보여줘.

GPT 기반 Agent:
1. XDR MCP 도구가 필요하다고 판단
2. get_incidents 또는 search_incidents 호출
3. 결과에서 severity, status, hostname 추출
4. 필요한 경우 TI DB와 cross-check
5. 최종 분석 답변 생성
```

여기서 GPT는 실제 XDR 데이터를 자체적으로 알고 있는 것이 아니다.

도구 호출을 통해 외부 시스템에서 데이터를 가져오고, 그 결과를 해석하는 역할을 한다.

---

## 20. GPT 구조를 이해할 때 자주 하는 오해

## 오해 1. GPT는 문장을 이해하고 한 번에 답변을 만든다

실제로는 다음 토큰을 반복 생성한다.

```text
다음 토큰 예측 → 선택 → 다음 토큰 예측 → 선택 → 반복
```

## 오해 2. GPT는 검색엔진처럼 정보를 찾는다

기본 GPT는 검색엔진이 아니다.

학습된 파라미터를 바탕으로 답을 생성한다.

최신 정보나 정확한 근거가 필요하면 RAG, 웹 검색, DB 조회가 필요하다.

## 오해 3. GPT는 항상 가장 확률 높은 토큰만 고른다

항상 그렇지는 않다.

temperature, top_p 같은 설정에 따라 다양한 토큰을 선택할 수 있다.

## 오해 4. GPT는 긴 문맥을 무한히 기억한다

그렇지 않다.

모델마다 context window 제한이 있다.

입력과 출력 토큰 수가 이 제한을 넘으면 일부 문맥은 사용할 수 없다.

---

## 21. GPT를 실무에서 볼 때 중요한 개념

GPT를 실제 시스템에 붙일 때는 다음 개념들이 중요하다.

```text
Context Window
Prompt
System Prompt
Token Limit
Output Token
Temperature
Top-p
Streaming
KV Cache
Function Calling
Tool Calling
Fine-tuning
RAG
```

이 중 특히 중요한 것은 Context Window다.

GPT는 입력된 모든 내용을 무제한으로 볼 수 있는 것이 아니다.

예를 들어 긴 로그, 긴 문서, 다수의 DB 조회 결과를 한 번에 넣으면 다음 문제가 생긴다.

```text
토큰 제한 초과
중요 정보 누락
응답 비용 증가
latency 증가
정확도 저하
```

그래서 실무에서는 다음 전략이 필요하다.

```text
문서 chunking
검색 결과 top-k 제한
요약 후 재입력
중요 필드만 추출
구조화된 JSON 사용
Agent 단계 분리
```

---

## 22. GPT 계열 발전 흐름

GPT 계열의 큰 흐름은 다음과 같다.

```text
GPT-1
→ GPT-2
→ GPT-3
→ InstructGPT
→ ChatGPT
→ GPT-4 계열
→ Multimodal / Tool-using / Reasoning Model 계열
```

각 단계의 의미를 간단히 보면 다음과 같다.

| 단계 | 핵심 의미 |
|---|---|
| GPT-1 | Transformer를 생성 모델에 적용 |
| GPT-2 | 더 큰 데이터와 모델 크기로 생성 능력 향상 |
| GPT-3 | 대규모 모델의 few-shot 능력 부각 |
| InstructGPT | 사람의 지시를 따르는 능력 강화 |
| ChatGPT | 대화형 인터페이스와 instruction following 대중화 |
| GPT-4 계열 | 추론, 멀티모달, 도구 활용 능력 강화 |

중요한 흐름은 단순히 모델 크기만 커진 것이 아니라는 점이다.

다음 요소들이 함께 발전했다.

```text
데이터 품질
모델 크기
학습 안정성
Instruction tuning
Preference alignment
Tool use
Serving infrastructure
Evaluation
Safety
```

---

## 23. 아주 단순한 GPT 생성 예시

개념적으로 GPT의 생성 루프는 다음과 비슷하다.

```python
text = "LLM은"

for step in range(50):
    tokens = tokenizer.encode(text)
    logits = model(tokens)
    next_token = sample(logits[-1])
    text += tokenizer.decode(next_token)
```

실제 구현은 훨씬 복잡하지만 핵심 구조는 이렇다.

```text
현재까지의 text를 토큰화한다.
모델이 다음 토큰 logits를 계산한다.
샘플링으로 다음 토큰을 고른다.
텍스트에 붙인다.
반복한다.
```

---

## 24. 네 프로젝트와 연결해서 이해하기

네가 하고 있는 MDR / RAG / Agent 프로젝트와 GPT 구조를 연결하면 다음과 같다.

| GPT 개념 | 네 프로젝트에서의 의미 |
|---|---|
| Tokenization | 사용자 질문, 로그, 문서가 토큰으로 변환됨 |
| Context Window | XDR 결과와 TI 결과를 모두 넣을 때 한계 발생 |
| Next Token Prediction | 최종 답변이 토큰 단위로 생성됨 |
| Causal Masking | 답변 생성 시 이전 문맥만 보고 다음 토큰 생성 |
| System Prompt | Agent 역할과 응답 규칙 정의 |
| Tool Calling | MCP, SQL Agent, 검색 도구 호출 판단 |
| RAG | 외부 보안 문서를 문맥으로 제공 |
| Streaming | 생성되는 토큰을 UI에 순차 전달 |
| KV Cache | 긴 응답 생성 시 이전 계산 재사용 |

즉 GPT 구조를 이해하면 왜 다음 문제가 생기는지도 더 잘 보인다.

```text
왜 긴 로그를 그대로 넣으면 답변 품질이 떨어지는가?
왜 RAG chunk 크기가 중요한가?
왜 tool result를 구조화해야 하는가?
왜 streaming은 토큰 생성 구조와 잘 맞는가?
왜 KV cache가 serving 성능에 중요한가?
```

---

## 25. 이번 편 핵심 요약

이번 편에서 가장 중요한 내용은 다음과 같다.

```text
GPT는 Decoder-only Transformer 구조를 사용한다.
GPT는 이전 토큰들을 보고 다음 토큰을 예측한다.
GPT는 Next Token Prediction 방식으로 학습된다.
GPT는 미래 토큰을 보지 못하도록 Causal Masking을 사용한다.
GPT의 답변 생성은 토큰 단위 반복 과정이다.
BERT는 이해 중심, GPT는 생성 중심이다.
RAG와 Agent는 GPT에 외부 문맥과 도구 사용 능력을 붙이는 구조다.
```

---

## 26. 스스로 점검해볼 질문

아래 질문에 답할 수 있으면 이번 편은 잘 이해한 것이다.

```text
1. GPT는 왜 Decoder-only 구조를 사용할까?
2. Next Token Prediction은 무엇인가?
3. Causal Masking은 왜 필요한가?
4. GPT와 BERT의 차이는 무엇인가?
5. GPT는 문장을 한 번에 생성하는가, 토큰 단위로 생성하는가?
6. Base Model과 Instruct Model은 무엇이 다른가?
7. RAG에서 GPT는 어떤 역할을 하는가?
8. Agent에서 GPT는 어떤 판단을 수행하는가?
```

---

## 27. 다음 편 예고

다음 편에서는 GPT 구조를 이해한 뒤 반드시 알아야 하는 주제로 넘어간다.

```text
8편: Tokenization과 Embedding
```

다음 편에서는 다음 내용을 다룬다.

```text
토큰이란 무엇인가?
왜 한글은 토큰 수가 많이 나올 수 있는가?
BPE, WordPiece, SentencePiece는 무엇인가?
Token ID와 Embedding Vector는 어떻게 연결되는가?
Context Window와 비용은 왜 토큰 기준으로 계산되는가?
```

LLM을 실무에서 제대로 쓰려면 Tokenization을 반드시 이해해야 한다.

