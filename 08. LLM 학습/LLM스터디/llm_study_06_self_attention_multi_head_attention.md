# LLM 공부 6편: Self-Attention과 Multi-Head Attention 깊게 보기

## 0. 이번 편의 목표

앞선 5편에서는 Transformer 전체 구조를 봤다.
이번 편에서는 Transformer의 가장 중요한 부품인 **Self-Attention**과 **Multi-Head Attention**을 조금 더 깊게 본다.

LLM을 제대로 이해하려면 결국 아래 질문에 답할 수 있어야 한다.

```text
문장 안의 각 토큰은 다른 토큰들과 어떻게 관계를 맺는가?
모델은 어떤 단어를 더 중요하게 봐야 하는지 어떻게 판단하는가?
Attention에서 Query, Key, Value는 각각 무엇인가?
Multi-Head Attention은 왜 여러 개의 attention을 병렬로 계산하는가?
```

이번 편의 핵심은 이거다.

```text
Self-Attention은 문장 안의 토큰들이 서로를 참고하면서
각 토큰의 의미 표현을 문맥에 맞게 바꾸는 메커니즘이다.
```

---

## 1. 왜 Self-Attention이 중요한가?

문장을 이해할 때 단어 하나만 따로 보면 의미가 부족하다.

예를 들어보자.

```text
나는 은행에 갔다.
나는 강가의 은행나무를 봤다.
```

여기서 `은행`이라는 단어는 문맥에 따라 의미가 달라진다.
첫 번째 문장의 은행은 금융기관일 가능성이 높고, 두 번째 문장에서는 은행나무를 의미한다.

즉, 단어의 의미는 고정된 것이 아니라 **주변 단어와의 관계**에 따라 달라진다.

기존 Word2Vec 같은 임베딩은 단어마다 어느 정도 고정된 벡터를 갖는다.  
하지만 Transformer의 Self-Attention은 같은 단어라도 문맥에 따라 다른 표현을 만들 수 있다.

```text
고정 임베딩:
은행 → 항상 비슷한 벡터

문맥 기반 임베딩:
은행 + 갔다 → 금융기관 의미에 가까운 벡터
은행 + 나무 → 식물 의미에 가까운 벡터
```

이 차이가 매우 중요하다.

LLM은 단어를 외워서 답하는 것이 아니라, 입력된 토큰들이 서로 어떤 관계를 가지는지 계속 계산하면서 문맥적 표현을 만든다.

---

## 2. Self-Attention의 직관

Self-Attention을 아주 단순하게 말하면 다음과 같다.

```text
각 토큰이 문장 안의 다른 토큰들을 바라보면서,
나에게 중요한 토큰은 더 많이 참고하고,
덜 중요한 토큰은 적게 참고하는 방식이다.
```

예를 들어 다음 문장을 보자.

```text
The animal didn't cross the street because it was too tired.
```

여기서 `it`은 무엇을 가리킬까?

```text
it → animal
```

모델은 `it`이라는 토큰을 처리할 때 `animal`, `street`, `tired` 같은 토큰과의 관계를 계산한다.
그리고 문맥상 `it`이 `animal`을 가리킨다고 판단할 수 있어야 한다.

Attention은 바로 이런 관계를 수치적으로 계산하는 구조다.

---

## 3. Self-Attention의 입력과 출력

Transformer에 들어가는 입력은 토큰 임베딩이다.

예를 들어 문장이 다음과 같다고 하자.

```text
I love security
```

토큰이 3개라면 각각의 토큰은 벡터로 표현된다.

```text
I        → x1
love     → x2
security → x3
```

Self-Attention은 이 벡터들을 받아서 새로운 벡터를 만든다.

```text
x1, x2, x3
   ↓ Self-Attention
z1, z2, z3
```

여기서 중요한 점은 `z1`, `z2`, `z3`가 단순히 자기 자신만 반영한 벡터가 아니라는 것이다.

각 출력 벡터는 다른 토큰들의 정보를 섞어서 만들어진다.

```text
z1 = I가 love와 security를 참고한 결과
z2 = love가 I와 security를 참고한 결과
z3 = security가 I와 love를 참고한 결과
```

즉, Self-Attention 이후의 각 토큰 표현은 **문맥이 반영된 표현**이 된다.

---

## 4. Query, Key, Value 개념

Self-Attention을 이해할 때 가장 많이 나오는 단어가 있다.

```text
Query
Key
Value
```

처음 보면 헷갈리지만, 검색 시스템에 비유하면 이해하기 쉽다.

```text
Query = 내가 찾고 싶은 것
Key   = 각 토큰이 가진 검색용 표지
Value = 실제로 가져올 정보
```

예를 들어 어떤 토큰이 다른 토큰들을 참고하려고 한다고 하자.

```text
현재 토큰: it
```

이 토큰은 문장 안에서 자신과 관련 있는 정보를 찾아야 한다.

이때 `it`은 Query를 만든다.

```text
Query: 나는 무엇과 관련이 있지?
```

다른 토큰들은 Key를 가지고 있다.

```text
animal → Key
street → Key
tired  → Key
```

Query와 Key를 비교해서 관련도가 높으면 해당 토큰의 Value를 많이 가져온다.

```text
Query(it)와 Key(animal)의 유사도 높음
→ Value(animal)를 많이 반영
```

정리하면 이렇다.

| 개념 | 역할 | 직관 |
|---|---|---|
| Query | 현재 토큰이 찾는 기준 | “나는 어떤 정보를 봐야 하지?” |
| Key | 각 토큰의 검색용 특징 | “나는 이런 정보를 가진 토큰이야” |
| Value | 실제로 가져올 내용 | “나를 참고하면 이 정보를 줄게” |

---

## 5. Q, K, V는 어디서 나오는가?

각 토큰 임베딩은 그대로 Query, Key, Value가 되는 것이 아니다.

입력 벡터에 각각 다른 가중치 행렬을 곱해서 Q, K, V를 만든다.

```text
입력 토큰 벡터 X

Q = XWq
K = XWk
V = XWv
```

여기서 `Wq`, `Wk`, `Wv`는 학습되는 파라미터다.

즉, 모델은 학습 과정에서 다음을 배운다.

```text
어떤 방식으로 Query를 만들지
어떤 방식으로 Key를 만들지
어떤 방식으로 Value를 만들지
```

이 점이 중요하다.

Attention은 사람이 직접 규칙을 넣는 방식이 아니다.
모델이 대규모 학습을 통해 “어떤 토큰을 어떻게 참고해야 하는지”를 스스로 학습한다.

---

## 6. Attention Score 계산

Self-Attention은 각 토큰 간의 관련도를 계산한다.

가장 기본적인 계산은 Query와 Key의 내적이다.

```text
Attention Score = Q · K
```

내적 값이 크다는 것은 두 벡터가 비슷한 방향을 가진다는 뜻이다.
즉, 두 토큰이 관련성이 높다고 볼 수 있다.

예를 들어 `it`이라는 토큰의 Query가 있고, 각 토큰의 Key가 있다고 하자.

```text
Query(it) · Key(animal) = 높음
Query(it) · Key(street) = 낮음
Query(it) · Key(tired)  = 중간
```

그러면 `it`은 `animal`을 더 많이 참고하게 된다.

---

## 7. Softmax로 가중치 만들기

Attention Score는 그대로 사용하지 않는다.

각 점수를 softmax에 넣어서 합이 1인 가중치로 바꾼다.

```text
score:  [5.0, 1.0, 2.0]
softmax → [0.94, 0.02, 0.04]
```

이렇게 되면 각 토큰을 얼마나 참고할지 비율이 나온다.

```text
animal: 94%
street: 2%
tired: 4%
```

이 가중치를 Value에 곱해서 최종 출력 벡터를 만든다.

```text
출력 = 0.94 × Value(animal)
     + 0.02 × Value(street)
     + 0.04 × Value(tired)
```

즉, Attention의 출력은 Value들의 가중합이다.

---

## 8. Scaled Dot-Product Attention

Transformer 논문에서 사용하는 Attention은 보통 다음 형태다.

```text
Attention(Q, K, V) = softmax(QKᵀ / √dk) V
```

처음 보면 복잡해 보이지만, 앞에서 본 과정을 한 줄로 쓴 것이다.

하나씩 보면 이렇다.

```text
QKᵀ       → Query와 Key의 관련도 계산
/ √dk     → 값이 너무 커지는 것을 방지하기 위한 스케일링
softmax   → 관련도를 확률처럼 정규화
× V       → 관련도에 따라 Value를 섞음
```

여기서 `dk`는 Key 벡터의 차원 수다.

왜 `√dk`로 나누는가?

벡터 차원이 커질수록 내적 값이 커질 가능성이 높다.
값이 너무 커지면 softmax 결과가 극단적으로 쏠릴 수 있다.
그러면 학습이 불안정해질 수 있다.

그래서 `√dk`로 나누어 값을 적당히 조절한다.

---

## 9. Self-Attention 계산 흐름 요약

전체 흐름을 정리하면 다음과 같다.

```text
1. 각 토큰을 임베딩 벡터로 변환한다.
2. 각 벡터에서 Q, K, V를 만든다.
3. Q와 K를 내적해서 토큰 간 관련도를 계산한다.
4. 관련도를 √dk로 나누어 스케일링한다.
5. softmax로 attention weight를 만든다.
6. attention weight를 V에 곱해 문맥 벡터를 만든다.
```

조금 더 도식화하면 다음과 같다.

```text
Token Embeddings
      ↓
Q, K, V 생성
      ↓
QKᵀ 계산
      ↓
Scaling
      ↓
Softmax
      ↓
Attention Weights
      ↓
Weighted Sum of V
      ↓
Contextual Token Representations
```

---

## 10. Self-Attention이 RNN보다 유리한 점

RNN은 문장을 순서대로 처리한다.

```text
x1 → x2 → x3 → x4 → x5
```

그래서 멀리 떨어진 단어 사이의 관계를 학습하기 어렵고, 병렬 처리도 어렵다.

반면 Self-Attention은 모든 토큰 간 관계를 한 번에 계산한다.

```text
x1 ↔ x2 ↔ x3 ↔ x4 ↔ x5
```

장점은 명확하다.

```text
1. 멀리 떨어진 토큰 간 관계를 직접 계산할 수 있다.
2. 문장 전체를 병렬로 처리할 수 있다.
3. 각 토큰이 문맥에 맞는 표현으로 바뀐다.
```

이게 Transformer가 대규모 학습에 강한 핵심 이유다.

---

## 11. 하지만 Self-Attention에도 비용이 있다

Self-Attention은 모든 토큰 쌍의 관계를 계산한다.

토큰 수가 `n`이면 Attention Score 행렬은 `n × n` 크기가 된다.

```text
토큰 10개  → 10 × 10 = 100
토큰 1,000개 → 1,000 × 1,000 = 1,000,000
토큰 100,000개 → 100,000 × 100,000 = 10,000,000,000
```

그래서 context window가 길어질수록 연산량과 메모리 사용량이 크게 증가한다.

이 문제 때문에 Long Context, Sparse Attention, FlashAttention, Sliding Window Attention 같은 기술들이 등장했다.

LLM을 운영하는 입장에서는 이 부분이 매우 중요하다.

```text
context window를 길게 준다고 항상 좋은 것은 아니다.
입력 토큰이 많아질수록 비용, latency, memory 사용량이 증가한다.
```

RAG에서 chunk를 너무 많이 넣으면 답변 품질이 오히려 나빠지거나 비용이 증가하는 이유도 여기에 연결된다.

---

## 12. Multi-Head Attention이 필요한 이유

Self-Attention 하나만 있으면 토큰 간 관계를 한 가지 관점으로만 본다.

하지만 문장 안의 관계는 하나가 아니다.

예를 들어 다음 문장을 보자.

```text
The security analyst investigated the incident because it looked suspicious.
```

여기에는 여러 관계가 있다.

```text
security analyst ↔ investigated  : 주어-동사 관계
incident ↔ suspicious             : 사건의 상태
it ↔ incident                      : 대명사 참조
security ↔ analyst                 : 명사 수식 관계
```

한 개의 Attention만으로는 이런 다양한 관계를 모두 잘 잡기 어렵다.
그래서 여러 개의 Attention을 병렬로 둔다.

이것이 Multi-Head Attention이다.

```text
Multi-Head Attention = 여러 개의 Attention Head가 서로 다른 관점에서 토큰 관계를 보는 구조
```

---

## 13. Head는 무엇인가?

Attention Head 하나는 독립적인 Q, K, V 변환을 가진다.

```text
Head 1: Q1, K1, V1
Head 2: Q2, K2, V2
Head 3: Q3, K3, V3
...
```

각 Head는 서로 다른 관계를 학습할 수 있다.

```text
Head 1 → 문법 관계에 집중
Head 2 → 대명사 참조에 집중
Head 3 → 위치 관계에 집중
Head 4 → 의미 관계에 집중
```

물론 실제 모델 내부의 Head가 반드시 사람이 해석하기 쉬운 방식으로 나뉘는 것은 아니다.
하지만 직관적으로는 이렇게 이해하면 된다.

```text
여러 관점에서 문장을 동시에 바라본다.
```

---

## 14. Multi-Head Attention 계산 흐름

Multi-Head Attention은 다음 순서로 동작한다.

```text
1. 입력 X에서 여러 벌의 Q, K, V를 만든다.
2. 각 Head에서 Attention을 독립적으로 계산한다.
3. 각 Head의 출력을 이어 붙인다.
4. 다시 선형 변환을 적용해 최종 출력을 만든다.
```

도식화하면 다음과 같다.

```text
Input X
  ├─ Head 1 Attention → output 1
  ├─ Head 2 Attention → output 2
  ├─ Head 3 Attention → output 3
  └─ Head 4 Attention → output 4
          ↓
      Concatenate
          ↓
    Linear Projection
          ↓
      Final Output
```

수식으로는 보통 이렇게 표현한다.

```text
MultiHead(Q, K, V) = Concat(head1, head2, ..., headh) Wo

headi = Attention(QWiQ, KWiK, VWiV)
```

여기서 중요한 것은 `WiQ`, `WiK`, `WiV`, `Wo` 모두 학습되는 파라미터라는 점이다.

---

## 15. 왜 Head를 여러 개로 나누는가?

하나의 큰 Attention을 쓰지 않고 여러 Head로 나누는 이유는 표현력을 높이기 위해서다.

예를 들어 512차원 벡터가 있다고 하자.
이를 8개 Head로 나누면 각 Head는 64차원씩 다른 관점의 Attention을 계산할 수 있다.

```text
전체 hidden size: 512
head 개수: 8
head dimension: 64
```

이렇게 하면 모델은 같은 문장을 여러 관점에서 분석할 수 있다.

```text
문법적 관계
의미적 관계
거리 관계
대명사 참조
문장 구조
주제 흐름
```

이런 다양한 관계를 병렬로 학습하는 것이 Multi-Head Attention의 핵심이다.

---

## 16. Encoder Self-Attention과 Decoder Self-Attention의 차이

Transformer에는 Encoder와 Decoder가 있다.
두 구조 모두 Self-Attention을 사용하지만 중요한 차이가 있다.

### Encoder Self-Attention

Encoder는 입력 문장 전체를 볼 수 있다.

```text
나는 오늘 보안 로그를 분석했다
```

각 토큰이 앞뒤 모든 토큰을 참고할 수 있다.

```text
현재 토큰 → 앞 토큰 참고 가능
현재 토큰 → 뒤 토큰 참고 가능
```

이런 방식은 BERT 같은 Encoder-only 모델에 사용된다.

### Decoder Self-Attention

GPT 같은 생성 모델은 미래 토큰을 보면 안 된다.

예를 들어 다음 토큰을 예측해야 한다고 하자.

```text
나는 오늘 보안
```

이 시점에서 모델은 뒤에 올 정답 토큰을 미리 보면 안 된다.
그래서 Decoder에서는 Causal Mask를 사용한다.

```text
현재 위치의 토큰은 자기 자신과 이전 토큰만 볼 수 있다.
미래 토큰은 볼 수 없다.
```

이 차이가 GPT를 이해할 때 매우 중요하다.

---

## 17. Causal Mask란 무엇인가?

Causal Mask는 미래 토큰을 가리는 장치다.

예를 들어 토큰이 4개 있다고 하자.

```text
토큰: t1 t2 t3 t4
```

각 위치에서 볼 수 있는 토큰은 다음과 같다.

```text
t1 → t1만 볼 수 있음
t2 → t1, t2 볼 수 있음
t3 → t1, t2, t3 볼 수 있음
t4 → t1, t2, t3, t4 볼 수 있음
```

행렬로 표현하면 다음과 같다.

```text
      t1  t2  t3  t4
t1    O   X   X   X
t2    O   O   X   X
t3    O   O   O   X
t4    O   O   O   O
```

`X`로 표시된 미래 위치는 attention score를 매우 작은 값으로 만들어 softmax 이후 거의 0이 되게 한다.

이렇게 해야 모델이 다음 토큰 예측을 정직하게 학습할 수 있다.

---

## 18. GPT에서 Self-Attention의 의미

GPT는 Decoder-only Transformer다.

즉, GPT의 Attention은 기본적으로 Causal Self-Attention이다.

```text
GPT = 미래 토큰을 보지 못하는 Self-Attention을 여러 층 쌓은 모델
```

GPT는 다음 토큰을 예측한다.

```text
입력: 보안 로그를 분석해보니
예측: "다음", "공격", "이상", ... 중 어떤 토큰이 올지 확률 계산
```

이때 각 토큰은 이전 토큰들을 참고하면서 문맥을 만든다.

```text
보안 → 이전 문맥 참고
로그 → 보안 참고
분석해보니 → 보안, 로그 참고
```

이 과정이 여러 layer를 거치면서 점점 더 복잡한 패턴을 학습한다.

---

## 19. Attention과 Context Window

LLM에서 context window는 모델이 한 번에 처리할 수 있는 토큰 범위다.

```text
context window = 모델이 attention으로 참고할 수 있는 최대 토큰 길이
```

예를 들어 context window가 8,000 tokens라면 모델은 최대 8,000개 토큰 범위 안에서 관계를 계산할 수 있다.

하지만 context window 안에 있다고 해서 모든 정보를 똑같이 잘 활용하는 것은 아니다.

실제로는 다음 문제가 생길 수 있다.

```text
입력이 너무 길면 중요한 정보가 묻힌다.
중간 위치의 정보가 약하게 반영될 수 있다.
불필요한 문맥이 많으면 답변이 흔들릴 수 있다.
연산 비용과 latency가 증가한다.
```

그래서 RAG에서는 단순히 많은 chunk를 넣는 것보다, 적절한 chunk를 선별해서 넣는 것이 중요하다.

---

## 20. Attention과 RAG의 연결

RAG에서 검색된 문서를 프롬프트에 넣으면 LLM은 해당 문서 토큰들을 attention으로 참고한다.

```text
사용자 질문 + 검색된 문서 chunk들
          ↓
LLM의 Self-Attention
          ↓
답변 생성
```

여기서 중요한 점은 검색된 문서가 context 안에 들어갔다고 해서 모델이 반드시 그 문서를 정확히 사용하는 것은 아니라는 것이다.

LLM은 attention을 통해 문서와 질문의 관련성을 계산하지만, 다음과 같은 문제가 생길 수 있다.

```text
검색된 chunk가 너무 많음
질문과 직접 관련 없는 chunk가 섞임
중요한 정보가 긴 문맥 중간에 묻힘
문서 간 내용이 충돌함
```

그래서 RAG 품질은 단순히 vector search만의 문제가 아니다.

```text
검색 품질
chunk 설계
reranking
prompt 구성
context 배치
모델의 attention 처리 능력
```

이 모든 것이 함께 작동한다.

---

## 21. Attention과 KV Cache

LLM serving에서 자주 나오는 개념이 KV Cache다.

GPT는 토큰을 하나씩 생성한다.

```text
입력 → 다음 토큰 생성 → 다시 입력에 붙임 → 다음 토큰 생성 → 반복
```

매번 전체 문맥에 대해 Key와 Value를 다시 계산하면 비효율적이다.
그래서 이전 토큰들의 Key와 Value를 저장해 둔다.

```text
KV Cache = 이전 토큰들의 Key, Value를 저장해 재사용하는 메모리 구조
```

생성 과정에서 새로 들어온 토큰의 Query만 계산하고, 이전 토큰들의 K/V는 cache에서 가져와 Attention을 계산할 수 있다.

이 덕분에 autoregressive generation이 훨씬 빨라진다.

vLLM의 PagedAttention 같은 기술도 이 KV Cache를 더 효율적으로 관리하기 위한 기술이라고 보면 된다.

---

## 22. Attention을 공부할 때 흔한 오해

### 오해 1. Attention weight가 곧 모델의 설명 가능성이다

Attention weight를 보면 어떤 토큰을 많이 참고했는지 알 수는 있다.
하지만 그것이 곧 모델이 왜 그런 답을 했는지에 대한 완전한 설명은 아니다.

LLM은 여러 layer, 여러 head, FFN, residual connection을 거친다.
그래서 attention만 보고 모델 판단 전체를 설명하기는 어렵다.

### 오해 2. Attention은 항상 사람이 이해할 수 있는 관계를 학습한다

일부 head는 사람이 보기에도 문법 관계나 대명사 참조처럼 해석될 수 있다.
하지만 모든 head가 사람이 이해하기 쉬운 의미를 갖는 것은 아니다.

### 오해 3. Context를 많이 넣으면 무조건 좋아진다

Context가 길어지면 모델이 참고할 수 있는 정보는 늘어난다.
하지만 불필요한 정보도 함께 늘어난다.

그래서 RAG에서는 “많이 넣기”보다 “정확한 정보를 잘 넣기”가 더 중요하다.

---

## 23. 실무 관점에서 꼭 기억할 점

LLM 시스템을 설계할 때 Attention은 다음 문제와 직접 연결된다.

```text
긴 프롬프트의 비용 증가
긴 context에서 답변 품질 저하
RAG chunk 개수 조절
KV cache 메모리 사용량
streaming 응답 latency
vLLM serving 최적화
multi-turn 대화의 history 관리
```

예를 들어 챗봇에서 대화 history를 계속 누적하면 context가 길어진다.
그러면 다음 문제가 생긴다.

```text
토큰 비용 증가
응답 속도 저하
중요한 최신 질문보다 과거 문맥이 섞임
KV cache 메모리 증가
```

그래서 실무에서는 history summarization, sliding window, retrieval-based memory 같은 방식이 필요하다.

---

## 24. 이번 편 핵심 요약

이번 편에서 반드시 가져가야 할 내용은 다음이다.

```text
Self-Attention은 각 토큰이 문장 안의 다른 토큰들을 참고해 문맥 기반 표현을 만드는 구조다.

Query는 현재 토큰이 찾는 기준이다.
Key는 각 토큰의 검색용 특징이다.
Value는 실제로 가져올 정보다.

Attention Score는 Query와 Key의 유사도로 계산된다.
Softmax를 통해 각 토큰을 얼마나 참고할지 가중치가 만들어진다.
최종 출력은 Value들의 가중합이다.

Multi-Head Attention은 여러 개의 attention head가 서로 다른 관점에서 토큰 관계를 학습하는 구조다.

GPT는 Causal Self-Attention을 사용하며, 미래 토큰을 볼 수 없도록 mask를 적용한다.

Context window가 길어질수록 attention 계산 비용과 메모리 사용량이 증가한다.

KV Cache는 이전 토큰들의 Key와 Value를 저장해 autoregressive generation을 빠르게 만드는 기술이다.
```

---

## 25. 다음 편 예고

다음 편에서는 GPT 계열 모델을 이해하기 위한 핵심 구조를 다룬다.

```text
7편: GPT와 Decoder-only Transformer
```

다음 편에서 다룰 내용은 다음과 같다.

```text
GPT는 왜 Decoder-only 구조를 쓰는가?
BERT와 GPT는 무엇이 다른가?
Next Token Prediction은 어떻게 학습되는가?
Causal Language Modeling이란 무엇인가?
Base Model과 Instruct Model은 무엇이 다른가?
```

이번 편까지 이해했다면 이제 LLM의 진짜 중심부로 들어갈 준비가 된 것이다.
