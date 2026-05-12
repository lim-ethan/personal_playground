# LLM 공부 5편: Transformer 구조 이해

## 0. 이번 편의 목표

이번 편에서는 LLM의 핵심 기반이 되는 **Transformer** 구조를 공부한다.

Transformer는 2017년 논문 **“Attention Is All You Need”**에서 제안된 구조다.  
이전까지 자연어처리 모델은 RNN, LSTM, GRU, Seq2Seq 같은 순차 처리 기반 구조가 중심이었다.

하지만 Transformer는 문장을 순서대로 하나씩 처리하지 않고, 문장 안의 토큰들이 서로 어떤 관계를 가지는지 **Attention**으로 한 번에 계산한다.

이번 편의 목표는 다음 질문에 답할 수 있게 되는 것이다.

```text
Transformer는 왜 등장했을까?
RNN/LSTM과 무엇이 다를까?
Self-Attention은 Transformer 안에서 어떤 역할을 할까?
Encoder와 Decoder는 무엇이 다를까?
왜 Transformer가 LLM 시대를 열었을까?
```

---

## 1. Transformer가 등장한 배경

이전 편에서 봤듯이 Seq2Seq 구조는 입력 문장을 고정된 벡터로 압축하고, Decoder가 그 벡터를 바탕으로 출력 문장을 생성했다.

하지만 이 방식에는 큰 한계가 있었다.

```text
입력 문장이 길어질수록 앞쪽 정보가 희미해진다.
모든 정보를 하나의 context vector에 압축하기 어렵다.
RNN 계열은 순차적으로 계산해야 해서 병렬 처리가 어렵다.
긴 문장에서 멀리 떨어진 단어 관계를 잡기 어렵다.
```

Attention은 이 문제 중 일부를 해결했다.

Decoder가 매 시점마다 Encoder의 모든 hidden state를 참고할 수 있게 되었기 때문이다.

하지만 여전히 Encoder와 Decoder 내부는 RNN 기반이었다.

즉, Attention이 들어갔지만 전체 구조는 여전히 순차 처리의 한계를 갖고 있었다.

Transformer는 여기서 더 과감한 선택을 한다.

```text
RNN을 아예 제거하고 Attention만으로 문장을 처리하자.
```

이 아이디어가 Transformer의 출발점이다.

---

## 2. Transformer의 핵심 아이디어

Transformer의 핵심은 간단하게 말하면 이렇다.

```text
문장 안의 모든 토큰이 서로를 직접 바라보게 하자.
```

예를 들어 다음 문장을 보자.

```text
The cat sat on the mat because it was tired.
```

여기서 `it`이 무엇을 가리키는지 이해하려면 앞쪽의 `cat`과 연결해야 한다.

RNN은 이런 관계를 순서대로 전달되는 hidden state를 통해 간접적으로 학습한다.

반면 Transformer는 Self-Attention을 통해 `it`이 문장 안의 다른 모든 단어를 직접 참고할 수 있다.

```text
it → The
it → cat
it → sat
it → on
it → the
it → mat
it → because
it → was
it → tired
```

그리고 모델은 이 중 어떤 단어가 중요한지 가중치로 계산한다.

이게 Self-Attention의 직관이다.

---

## 3. Transformer 전체 구조

원래 Transformer는 크게 두 부분으로 구성된다.

```text
Encoder
Decoder
```

전체 구조를 단순화하면 다음과 같다.

```text
입력 문장
  ↓
Token Embedding
  ↓
Positional Encoding
  ↓
Encoder Stack
  ↓
Decoder Stack
  ↓
Linear + Softmax
  ↓
출력 토큰
```

기계번역 예시로 보면 다음과 같다.

```text
입력: I love you
출력: 나는 너를 사랑해
```

Encoder는 입력 문장을 이해하고, Decoder는 그 이해 결과를 바탕으로 출력 문장을 생성한다.

---

## 4. Encoder의 역할

Encoder는 입력 문장을 읽고, 각 토큰의 문맥적 표현을 만든다.

예를 들어 `bank`라는 단어가 있다고 하자.

```text
I deposited money at the bank.
I sat near the river bank.
```

두 문장에서 `bank`는 철자는 같지만 의미가 다르다.

Encoder는 주변 단어를 참고해서 `bank`가 금융기관인지 강둑인지 구분할 수 있는 벡터 표현을 만든다.

Encoder 한 층은 보통 다음 구성으로 되어 있다.

```text
Multi-Head Self-Attention
→ Add & Norm
→ Feed Forward Network
→ Add & Norm
```

이 구조가 여러 층 반복된다.

```text
Encoder Layer 1
Encoder Layer 2
Encoder Layer 3
...
Encoder Layer N
```

BERT는 이 Encoder 구조를 중심으로 만들어진 모델이다.

---

## 5. Decoder의 역할

Decoder는 출력 문장을 생성하는 역할을 한다.

기계번역에서는 Encoder가 입력 문장을 이해하고, Decoder가 번역 문장을 한 토큰씩 생성한다.

Decoder 한 층은 보통 다음 구성으로 되어 있다.

```text
Masked Multi-Head Self-Attention
→ Add & Norm
→ Encoder-Decoder Attention
→ Add & Norm
→ Feed Forward Network
→ Add & Norm
```

여기서 중요한 것은 **Masked Self-Attention**이다.

Decoder는 문장을 생성할 때 미래 토큰을 볼 수 없어야 한다.

예를 들어 다음 문장을 생성한다고 하자.

```text
나는 밥을 먹었다
```

`나는`을 생성하는 시점에는 뒤에 올 `밥을`, `먹었다`를 미리 보면 안 된다.

그래서 Decoder에서는 미래 위치를 가리는 mask를 사용한다.

```text
현재 토큰은 이전 토큰만 볼 수 있다.
미래 토큰은 볼 수 없다.
```

GPT 계열 모델은 이 Decoder 구조를 중심으로 만들어졌다.

정확히는 Encoder-Decoder Attention 없이, Decoder-only Transformer 구조를 사용한다.

---

## 6. Self-Attention 다시 보기

Transformer의 핵심은 Self-Attention이다.

Self-Attention은 문장 안의 각 토큰이 다른 토큰을 얼마나 참고해야 하는지 계산한다.

예를 들어 다음 문장을 보자.

```text
The animal didn't cross the street because it was too tired.
```

여기서 `it`은 `animal`을 가리킨다.

Self-Attention은 `it`이라는 토큰을 처리할 때 `animal`에 높은 attention weight를 줄 수 있다.

단순화하면 이런 과정이다.

```text
각 토큰을 벡터로 바꾼다.
각 토큰이 다른 토큰과 얼마나 관련 있는지 점수를 계산한다.
점수를 softmax로 확률처럼 바꾼다.
그 가중치를 이용해 다른 토큰 정보를 섞는다.
```

결과적으로 각 토큰은 자기 자신만의 의미가 아니라, 문장 전체 문맥이 반영된 표현으로 바뀐다.

---

## 7. Query, Key, Value 직관

Self-Attention을 공부할 때 가장 많이 나오는 개념이 Q, K, V다.

```text
Q = Query
K = Key
V = Value
```

처음 보면 헷갈리지만, 검색 시스템에 비유하면 쉽다.

```text
Query: 내가 찾고 싶은 것
Key: 각 정보의 색인 또는 특징
Value: 실제로 가져올 정보
```

예를 들어 `it`이라는 토큰이 어떤 단어를 참고해야 하는지 찾는다고 하자.

```text
it의 Query와 다른 단어들의 Key를 비교한다.
비교 점수가 높으면 더 많이 참고한다.
참고할 때는 해당 단어의 Value를 가져온다.
```

즉, Attention은 다음과 같은 흐름이다.

```text
Query와 Key를 비교해서 중요도를 계산하고,
그 중요도만큼 Value를 섞는다.
```

Transformer에서는 입력 토큰 벡터를 각각 다른 가중치 행렬에 통과시켜 Q, K, V를 만든다.

```text
입력 벡터 X
  ↓
Wq를 곱함 → Q
Wk를 곱함 → K
Wv를 곱함 → V
```

---

## 8. Scaled Dot-Product Attention

Transformer에서 Attention 계산은 보통 다음 형태로 표현된다.

```text
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

처음 보면 수식이 부담스럽지만 의미는 단순하다.

```text
QKᵀ: Query와 Key의 유사도 계산
/ √d_k: 값이 너무 커지지 않도록 스케일 조정
softmax: 중요도를 확률처럼 변환
× V: 중요도에 따라 Value를 섞음
```

즉, 수식을 말로 풀면 이렇다.

```text
각 토큰이 다른 토큰을 얼마나 봐야 하는지 계산하고,
그 비율대로 정보를 가져와서 새로운 표현을 만든다.
```

여기서 중요한 건 수식을 외우는 게 아니다.

중요한 건 다음 직관이다.

```text
Attention은 토큰 간 관계를 동적으로 계산하는 장치다.
```

---

## 9. Multi-Head Attention

Transformer는 Attention을 한 번만 계산하지 않는다.

여러 개의 Attention head를 병렬로 사용한다.

이를 **Multi-Head Attention**이라고 한다.

왜 여러 head가 필요할까?

문장에는 여러 종류의 관계가 있기 때문이다.

예를 들어 한 head는 주어-동사 관계를 볼 수 있고, 다른 head는 대명사 참조 관계를 볼 수 있다.

또 다른 head는 위치적 관계나 구문 구조를 볼 수 있다.

```text
Head 1: 주어-동사 관계
Head 2: 대명사 참조
Head 3: 수식어 관계
Head 4: 문장 구조
```

물론 실제로 각 head가 이렇게 명확하게 역할이 나뉘는 것은 아니다.

하지만 직관적으로는 여러 관점에서 문장을 바라보는 구조라고 이해하면 된다.

Multi-Head Attention 흐름은 다음과 같다.

```text
입력 벡터
  ↓
여러 개의 Q, K, V로 분리
  ↓
각 head에서 Attention 계산
  ↓
결과를 concat
  ↓
Linear projection
```

---

## 10. Positional Encoding

Transformer는 RNN처럼 순서대로 토큰을 처리하지 않는다.

그래서 별도의 위치 정보가 필요하다.

예를 들어 다음 두 문장은 단어는 같지만 의미가 다르다.

```text
Dog bites man.
Man bites dog.
```

단어 집합만 보면 둘 다 `Dog`, `bites`, `man`으로 구성되어 있다.

하지만 순서가 달라서 의미가 완전히 달라진다.

Transformer는 토큰을 병렬로 처리하기 때문에, 입력 임베딩에 위치 정보를 더해줘야 한다.

이를 **Positional Encoding**이라고 한다.

```text
Token Embedding + Positional Encoding = 최종 입력 표현
```

초기 Transformer 논문에서는 sin, cos 함수를 이용한 고정 위치 인코딩을 사용했다.

이후 모델들에서는 learned positional embedding, RoPE, ALiBi 같은 다양한 방식이 등장했다.

LLM을 깊게 공부하다 보면 RoPE는 꼭 다시 보게 된다.

특히 Llama, Qwen 같은 현대 Decoder-only 모델을 이해할 때 중요하다.

---

## 11. Feed Forward Network

Attention이 토큰 간 관계를 섞는 역할이라면, Feed Forward Network는 각 토큰 표현을 비선형적으로 변환하는 역할을 한다.

Encoder나 Decoder의 각 layer에는 Attention 뒤에 FFN이 들어간다.

```text
Attention: 토큰 간 정보를 섞음
FFN: 각 토큰의 표현을 더 풍부하게 변환함
```

일반적으로 FFN은 다음과 같은 구조다.

```text
Linear
→ Activation
→ Linear
```

각 토큰 위치마다 같은 FFN이 적용된다.

즉, 문장 전체를 한 번에 섞는 것이 아니라, Attention으로 섞인 각 토큰 벡터를 개별적으로 변환한다.

---

## 12. Residual Connection과 Layer Normalization

Transformer는 여러 층을 깊게 쌓는다.

층이 깊어지면 학습이 불안정해질 수 있다.

이를 완화하기 위해 Transformer는 Residual Connection과 Layer Normalization을 사용한다.

### Residual Connection

Residual Connection은 입력을 출력에 더해주는 구조다.

```text
출력 = 어떤 변환 함수(x) + x
```

쉽게 말하면 모델이 새로운 정보를 학습하되, 기존 정보도 잃지 않도록 도와준다.

이 구조 덕분에 깊은 네트워크를 더 안정적으로 학습할 수 있다.

### Layer Normalization

Layer Normalization은 각 층의 값 분포를 안정화한다.

학습 중 값이 너무 커지거나 작아지는 문제를 완화한다.

Transformer layer를 보면 보통 다음 패턴이 반복된다.

```text
Sub-layer
→ Add
→ Norm
```

또는 현대 LLM에서는 Pre-LN 구조처럼 순서가 약간 다르게 쓰이기도 한다.

```text
Norm
→ Sub-layer
→ Add
```

실무적으로는 Pre-LN 구조가 깊은 모델 학습에 더 안정적이라 많이 사용된다.

---

## 13. Encoder-only, Decoder-only, Encoder-Decoder

Transformer 기반 모델은 크게 세 종류로 나눌 수 있다.

```text
Encoder-only
Decoder-only
Encoder-Decoder
```

### Encoder-only

대표 모델은 BERT다.

```text
입력 문장을 이해하는 데 강함
분류, 검색, 임베딩, 문장 유사도에 많이 사용
```

BERT는 양방향 문맥을 본다.

즉, 어떤 단어를 이해할 때 앞뒤 단어를 모두 참고할 수 있다.

### Decoder-only

대표 모델은 GPT다.

```text
텍스트 생성에 강함
다음 토큰 예측 방식으로 학습
챗봇, 코드 생성, 요약, Agent 등에 사용
```

Decoder-only 모델은 왼쪽에서 오른쪽으로 다음 토큰을 생성한다.

현재 대부분의 LLM 서비스는 Decoder-only 구조를 기반으로 한다.

### Encoder-Decoder

대표 모델은 원래 Transformer, T5, BART 등이 있다.

```text
입력을 이해하고 출력을 생성하는 구조
번역, 요약, 변환 작업에 강함
```

입력과 출력이 명확하게 구분되는 작업에 적합하다.

---

## 14. Transformer와 GPT의 관계

GPT는 Transformer의 Decoder 구조를 기반으로 한다.

하지만 원래 Transformer Decoder에 있던 Encoder-Decoder Attention은 사용하지 않는다.

왜냐하면 GPT는 별도의 Encoder 입력을 받는 구조가 아니라, 지금까지의 토큰만 보고 다음 토큰을 예측하기 때문이다.

GPT의 단순화된 구조는 다음과 같다.

```text
입력 토큰
  ↓
Token Embedding
  ↓
Position Embedding
  ↓
Masked Self-Attention
  ↓
Feed Forward Network
  ↓
여러 layer 반복
  ↓
다음 토큰 확률 출력
```

즉, GPT는 다음 구조라고 볼 수 있다.

```text
Decoder-only Transformer
+ Causal Masking
+ Next Token Prediction
```

이게 LLM의 가장 핵심적인 구조다.

---

## 15. 왜 Transformer가 LLM 시대를 열었나?

Transformer가 중요한 이유는 단순히 성능이 좋아서가 아니다.

대규모 학습에 매우 적합했기 때문이다.

RNN 계열은 문장을 순서대로 처리해야 해서 병렬화가 어렵다.

반면 Transformer는 문장 안의 토큰 관계를 행렬 연산으로 한 번에 계산할 수 있다.

이 말은 GPU를 훨씬 효율적으로 사용할 수 있다는 뜻이다.

```text
RNN/LSTM:
토큰을 순서대로 처리해야 함
병렬화 어려움
긴 문장 처리 한계

Transformer:
토큰 간 관계를 병렬 계산
GPU 활용에 적합
대규모 데이터와 모델 확장에 유리
```

LLM 시대를 만든 핵심 요인은 다음 세 가지다.

```text
1. Transformer 구조
2. 대규모 텍스트 데이터
3. GPU/TPU 기반 대규모 병렬 학습
```

이 세 가지가 결합되면서 GPT-2, GPT-3, ChatGPT 같은 모델이 등장할 수 있었다.

---

## 16. Transformer를 공부할 때 자주 헷갈리는 부분

### Attention과 Self-Attention은 같은가?

완전히 같은 말은 아니다.

Attention은 일반적인 메커니즘이다.

Self-Attention은 Query, Key, Value가 모두 같은 입력에서 나오는 Attention이다.

```text
Attention: 서로 다른 입력 사이 관계를 볼 수도 있음
Self-Attention: 같은 문장 내부 토큰끼리 관계를 봄
```

### Transformer는 RNN을 완전히 대체했나?

LLM과 현대 NLP에서는 대부분 Transformer가 중심이다.

하지만 RNN 계열이 완전히 사라진 것은 아니다.

시계열, 임베디드 환경, 경량 모델 등에서는 여전히 쓰일 수 있다.

다만 대규모 언어모델의 중심 구조는 Transformer가 맞다.

### GPT는 Transformer Encoder를 쓰나?

아니다.

GPT는 Encoder가 아니라 Decoder-only 구조다.

다만 원래 Transformer Decoder와 완전히 동일한 구조는 아니고, LLM에 맞게 변형된 Decoder-only 구조라고 이해하면 된다.

---

## 17. 실무 관점에서 Transformer 이해가 중요한 이유

Transformer 구조를 이해하면 LLM을 쓸 때 많은 개념이 연결된다.

예를 들어 다음 개념들이 모두 Transformer와 연결된다.

```text
context window
KV cache
attention cost
long context model
token limit
inference latency
batching
vLLM serving
LoRA fine-tuning
quantization
```

특히 LLM serving을 할 때는 Attention 구조를 이해해야 한다.

왜냐하면 긴 context를 넣을수록 Attention 계산 비용이 커지고, KV Cache 메모리도 증가하기 때문이다.

간단히 말하면 다음과 같다.

```text
입력 토큰이 길어질수록 계산량과 메모리 사용량이 증가한다.
```

네가 vLLM, Ollama, GPU 메모리, RAG chunking을 다루고 있다면 Transformer 구조 이해는 선택이 아니라 필수에 가깝다.

---

## 18. 이번 편 핵심 요약

이번 편의 핵심은 다음과 같다.

```text
Transformer는 RNN 없이 Attention만으로 문장을 처리하는 구조다.
```

조금 더 풀면 다음과 같다.

```text
RNN/LSTM은 순차 처리 기반이라 병렬화와 긴 문장 처리에 한계가 있었다.
Attention은 중요한 입력 위치를 직접 참고할 수 있게 했다.
Transformer는 RNN을 제거하고 Self-Attention 중심 구조를 만들었다.
Self-Attention은 문장 안의 토큰들이 서로를 직접 참고하게 한다.
Multi-Head Attention은 여러 관점에서 토큰 관계를 본다.
Positional Encoding은 순서 정보를 제공한다.
Encoder는 이해에 강하고, Decoder는 생성에 강하다.
GPT는 Decoder-only Transformer 기반이다.
Transformer는 GPU 병렬화와 대규모 학습에 적합해서 LLM 시대를 열었다.
```

---

## 19. 꼭 기억해야 할 문장

```text
Transformer는 모든 토큰이 서로를 직접 바라보는 구조다.
```

```text
Self-Attention은 토큰 간 관계를 동적으로 계산한다.
```

```text
GPT는 Decoder-only Transformer로 다음 토큰을 예측한다.
```

```text
LLM의 성능과 한계는 대부분 Transformer 구조와 연결되어 있다.
```

---

## 20. 다음 편 예고

다음 편에서는 Transformer 안에서도 가장 중요한 부분인 **Self-Attention과 Multi-Head Attention**을 더 깊게 본다.

이번 편에서는 구조 전체를 봤다면, 다음 편에서는 다음 내용을 집중적으로 다룬다.

```text
Q, K, V는 실제로 무엇인가?
Attention score는 어떻게 계산되는가?
softmax는 왜 들어가는가?
Multi-Head는 왜 필요한가?
Causal Mask는 GPT에서 왜 중요한가?
Attention 계산 비용은 왜 문제가 되는가?
```

다음 편 제목은 다음과 같이 잡으면 좋다.

```text
LLM 공부 6편: Self-Attention과 Multi-Head Attention 깊게 보기
```
