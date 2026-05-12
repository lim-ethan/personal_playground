# LLM 공부 4편: Attention

## 0. 이번 편의 목표

이번 편에서는 Transformer로 넘어가기 직전의 핵심 개념인 **Attention**을 다룬다.

Attention은 LLM 역사에서 매우 중요한 전환점이다.  
이전 편에서 본 RNN, LSTM, Seq2Seq는 문장을 순서대로 처리하면서 문맥을 압축하려고 했다. 하지만 입력 문장이 길어질수록 중요한 정보가 희미해지는 문제가 있었다.

Attention은 이 문제를 해결하기 위해 등장했다.

이번 편을 끝까지 보면 아래 질문에 답할 수 있어야 한다.

```text
Seq2Seq 구조의 가장 큰 병목은 무엇이었나?
Attention은 왜 필요한가?
Attention에서 Query, Key, Value는 무엇인가?
Attention score는 어떤 의미인가?
Transformer의 Self-Attention과 어떤 관계가 있는가?
```

---

## 1. Attention이 등장한 배경

Seq2Seq 모델은 입력 문장을 하나의 고정된 벡터로 압축한 뒤, 그 벡터를 기반으로 출력 문장을 생성했다.

예를 들어 번역 모델을 생각해보자.

```text
입력 문장: 나는 어제 친구와 서울에서 점심을 먹었다
출력 문장: I had lunch with a friend in Seoul yesterday
```

기존 Seq2Seq 구조에서는 Encoder가 입력 문장을 읽고, 마지막 hidden state 하나에 전체 의미를 담으려고 했다.

```text
입력 문장 전체
→ Encoder
→ 하나의 context vector
→ Decoder
→ 출력 문장 생성
```

문장이 짧으면 어느 정도 가능하다.  
하지만 문장이 길어지면 문제가 생긴다.

---

## 2. Seq2Seq의 병목: 고정 길이 Context Vector

기존 Seq2Seq 구조의 핵심 문제는 **모든 입력 정보를 하나의 벡터에 압축해야 한다는 것**이다.

```text
긴 입력 문장
→ 하나의 context vector에 압축
→ Decoder가 이 벡터만 보고 번역
```

이 방식은 마치 긴 회의 내용을 한 문장으로 요약한 뒤, 그 요약문만 보고 전체 회의록을 다시 복원하려는 것과 비슷하다.

당연히 세부 정보가 손실된다.

특히 아래와 같은 문제가 발생한다.

```text
1. 입력 문장이 길어질수록 앞부분 정보가 약해진다.
2. Decoder는 매 시점마다 같은 context vector에만 의존한다.
3. 출력 단어마다 참고해야 할 입력 단어가 다른데, 이를 반영하기 어렵다.
```

예를 들어 영어로 번역할 때, 출력 단어 `Seoul`을 생성하는 순간에는 입력 문장의 `서울에서`를 집중적으로 봐야 한다.  
하지만 기존 Seq2Seq에서는 Decoder가 입력 전체를 압축한 하나의 벡터만 받기 때문에 특정 단어에 집중하기 어렵다.

---

## 3. Attention의 핵심 아이디어

Attention의 아이디어는 단순하다.

> 출력 단어를 생성할 때마다 입력 문장의 어느 부분을 더 중요하게 볼지 결정하자.

즉, Decoder가 매번 Encoder의 모든 hidden state를 다시 참고한다.

기존 Seq2Seq는 마지막 context vector 하나만 사용했다.

```text
Encoder 마지막 hidden state 하나만 사용
```

Attention은 Encoder의 모든 hidden state를 사용한다.

```text
Encoder hidden state 전체 사용
```

예를 들어 입력 문장이 다음과 같다고 하자.

```text
나는 / 어제 / 친구와 / 서울에서 / 점심을 / 먹었다
```

Decoder가 `Seoul`을 생성하려는 순간에는 `서울에서`에 높은 가중치를 준다.

Decoder가 `yesterday`를 생성하려는 순간에는 `어제`에 높은 가중치를 준다.

Decoder가 `lunch`를 생성하려는 순간에는 `점심을`에 높은 가중치를 준다.

이렇게 출력 시점마다 입력 단어별 중요도를 다르게 계산하는 것이 Attention이다.

---

## 4. Attention을 직관적으로 이해하기

Attention은 검색 과정과 비슷하게 볼 수 있다.

사용자가 어떤 질문을 한다.

```text
질문: 출력할 다음 단어를 만들기 위해 지금 어떤 입력 정보를 봐야 하지?
```

모델은 입력 문장 전체를 훑으면서 관련성이 높은 부분을 찾는다.

```text
입력 단어 1: 나는       관련도 낮음
입력 단어 2: 어제       관련도 중간
입력 단어 3: 친구와     관련도 낮음
입력 단어 4: 서울에서   관련도 높음
입력 단어 5: 점심을     관련도 낮음
입력 단어 6: 먹었다     관련도 낮음
```

그리고 관련도가 높은 단어의 정보를 더 많이 반영해서 다음 출력을 만든다.

즉, Attention은 다음과 같은 과정이다.

```text
1. 현재 출력 시점에서 필요한 정보가 무엇인지 판단한다.
2. 입력 문장의 각 위치와 관련도를 계산한다.
3. 관련도가 높은 위치에 높은 가중치를 준다.
4. 가중합을 만들어 Decoder에 전달한다.
```

---

## 5. Attention Score

Attention에서는 입력의 각 위치가 현재 출력 시점과 얼마나 관련 있는지 점수로 계산한다.

이 점수를 **Attention score**라고 한다.

예를 들어 Decoder가 `Seoul`을 만들고 싶다고 하자.

```text
입력 토큰       Attention score
나는           0.02
어제           0.05
친구와         0.03
서울에서       0.80
점심을         0.06
먹었다         0.04
```

이 경우 모델은 `서울에서`를 가장 중요하게 본다.

Attention score는 보통 softmax를 거쳐 확률처럼 정규화된다.

```text
전체 attention score의 합 = 1
```

이렇게 하면 입력의 각 위치에 대한 중요도를 비율로 표현할 수 있다.

---

## 6. Attention의 기본 계산 흐름

Attention은 크게 세 단계로 볼 수 있다.

```text
1. 현재 Decoder 상태와 Encoder hidden state들의 유사도를 계산한다.
2. 유사도 점수를 softmax로 정규화한다.
3. 정규화된 가중치로 Encoder hidden state들을 가중합한다.
```

조금 더 구조적으로 쓰면 다음과 같다.

```text
Decoder 현재 상태
        ↓
Encoder 각 hidden state와 비교
        ↓
Attention score 계산
        ↓
Softmax로 attention weight 계산
        ↓
Encoder hidden state들의 weighted sum 생성
        ↓
Decoder가 다음 단어 생성에 사용
```

여기서 weighted sum이 바로 Attention이 만든 새로운 context vector다.

중요한 점은 이 context vector가 매 출력 시점마다 달라진다는 것이다.

기존 Seq2Seq:

```text
입력 문장 전체 → 고정 context vector 1개
```

Attention 기반 Seq2Seq:

```text
출력 시점마다 다른 context vector 생성
```

---

## 7. Query, Key, Value의 직관

Attention을 공부하다 보면 Query, Key, Value라는 개념이 나온다.

처음 보면 추상적이지만, 검색 시스템으로 생각하면 이해하기 쉽다.

```text
Query: 내가 찾고 싶은 것
Key: 각 정보가 어떤 내용인지 나타내는 색인
Value: 실제로 가져올 정보
```

예를 들어 도서관 검색을 생각해보자.

```text
Query: Transformer를 설명하는 책을 찾고 싶다
Key: 책 제목, 키워드, 분류 정보
Value: 책 본문 내용
```

검색 과정은 다음과 같다.

```text
Query와 Key를 비교해서 관련도를 계산한다.
관련도가 높은 Key에 해당하는 Value를 가져온다.
```

Attention도 똑같다.

```text
Query와 Key의 유사도 계산
→ Attention weight 생성
→ Value를 weighted sum
```

즉, Attention은 다음 공식의 직관적 표현이다.

```text
필요한 정보(Query)를 기준으로
각 후보 정보(Key)와의 관련도를 계산하고
실제 정보(Value)를 가중합해서 가져온다.
```

---

## 8. Query, Key, Value를 번역 예시로 보기

번역 모델에서 Decoder가 다음 단어를 생성하려고 한다.

```text
현재 Decoder가 만들려는 단어: Seoul
```

이때 Query는 현재 Decoder의 상태다.

```text
Query = 지금 출력하려는 위치에서 필요한 정보
```

Encoder의 각 입력 토큰은 Key와 Value를 가진다.

```text
입력 토큰: 나는       Key, Value
입력 토큰: 어제       Key, Value
입력 토큰: 친구와     Key, Value
입력 토큰: 서울에서   Key, Value
입력 토큰: 점심을     Key, Value
입력 토큰: 먹었다     Key, Value
```

모델은 Query와 각 Key를 비교한다.

```text
Query와 '서울에서'의 Key가 가장 잘 맞음
```

그러면 `서울에서`의 Value가 크게 반영된다.

```text
Attention output ≈ 서울에서 관련 정보가 많이 반영된 벡터
```

이 벡터를 기반으로 Decoder는 `Seoul`이라는 단어를 생성할 가능성이 높아진다.

---

## 9. Attention이 해결한 문제

Attention은 기존 Seq2Seq의 중요한 한계를 해결했다.

### 9.1 긴 문장 처리 개선

하나의 고정 벡터에 전체 문장을 압축하지 않아도 된다.

Decoder는 필요할 때마다 Encoder의 모든 hidden state를 참고할 수 있다.

```text
기존 방식: 전체 정보를 하나의 벡터에 압축
Attention: 필요한 위치를 동적으로 참고
```

### 9.2 정렬 문제 해결

번역에서는 입력 단어와 출력 단어가 대응되는 경우가 많다.

```text
서울에서 → Seoul
어제 → yesterday
점심 → lunch
```

Attention은 출력 단어를 만들 때 입력의 어느 부분을 봤는지 가중치로 보여준다.  
그래서 단어 간 대응 관계를 더 잘 학습할 수 있다.

### 9.3 해석 가능성 일부 제공

Attention weight를 보면 모델이 어느 입력 위치를 중요하게 봤는지 어느 정도 확인할 수 있다.

물론 Attention weight가 항상 완벽한 설명은 아니다.  
하지만 기존 RNN/LSTM 구조보다는 내부 동작을 해석할 실마리를 더 많이 제공했다.

---

## 10. Attention과 Self-Attention의 차이

여기서 중요한 연결점이 있다.

초기 Attention은 주로 Encoder-Decoder 구조에서 사용됐다.

```text
Decoder가 Encoder의 hidden state를 참고한다.
```

즉, 출력 쪽이 입력 쪽을 바라보는 구조다.

하지만 Transformer에서 핵심이 되는 것은 **Self-Attention**이다.

Self-Attention은 문장 내부의 토큰들이 서로를 바라보는 구조다.

```text
문장: 나는 어제 서울에서 점심을 먹었다

'나는'     → 다른 토큰들을 참고
'어제'     → 다른 토큰들을 참고
'서울에서' → 다른 토큰들을 참고
'점심을'   → 다른 토큰들을 참고
'먹었다'   → 다른 토큰들을 참고
```

즉, Self-Attention에서는 Query, Key, Value가 모두 같은 입력 문장에서 나온다.

```text
일반 Attention:
Query는 Decoder에서, Key/Value는 Encoder에서 나올 수 있음

Self-Attention:
Query, Key, Value가 모두 같은 문장 내부에서 나옴
```

이 Self-Attention이 Transformer의 중심 개념이다.

---

## 11. 왜 Attention이 Transformer로 이어졌나?

RNN/LSTM은 순차적으로 계산한다.

```text
토큰 1 → 토큰 2 → 토큰 3 → 토큰 4
```

이 구조는 문장 순서를 자연스럽게 반영하지만, 병렬 처리가 어렵다.

반면 Attention은 문장 내 모든 토큰 간의 관계를 한 번에 계산할 수 있다.

```text
모든 토큰 쌍의 관련도 계산
```

이 특성 덕분에 Transformer는 RNN 없이도 문맥을 처리할 수 있게 됐다.

Transformer의 핵심 아이디어는 다음과 같다.

```text
RNN을 제거하고 Attention만으로 문맥을 처리하자.
```

이것이 2017년 논문 제목인 **Attention Is All You Need**의 의미다.

즉, Attention은 단순한 보조 기법이 아니라 LLM 시대를 연 핵심 기술이다.

---

## 12. Attention을 LLM 관점에서 다시 보기

현대 LLM은 대부분 Transformer 기반이다.  
그리고 Transformer의 핵심은 Self-Attention이다.

LLM이 긴 질문을 읽고 답변할 수 있는 이유도 결국 입력 토큰들 사이의 관계를 Attention으로 계산하기 때문이다.

예를 들어 사용자가 이렇게 질문했다고 하자.

```text
최근 7일 동안 high severity incident 목록을 보여주고, 관련 asset과 user도 같이 정리해줘.
```

모델은 다음 관계를 이해해야 한다.

```text
최근 7일 ↔ 시간 조건
high severity ↔ severity 조건
incident 목록 ↔ 조회 대상
asset과 user ↔ 함께 출력할 필드
```

Self-Attention은 이런 토큰 간 관계를 계산하는 데 사용된다.

LLM이 문장 전체를 단순히 왼쪽에서 오른쪽으로만 읽는 것이 아니라, 내부적으로 토큰들 사이의 관련성을 계속 계산한다고 보면 된다.

---

## 13. 간단한 예시: Attention Weight

다음 문장을 생각해보자.

```text
The animal didn't cross the street because it was too tired.
```

여기서 `it`은 무엇을 가리킬까?

```text
it = the animal
```

모델이 `it`을 이해하려면 앞의 `animal`과 연결해야 한다.

Attention 관점에서는 `it` 토큰이 `animal` 토큰에 높은 attention weight를 줄 수 있다.

반대로 다음 문장을 보자.

```text
The animal didn't cross the street because it was too wide.
```

여기서 `it`은 무엇을 가리킬까?

```text
it = the street
```

이번에는 `it`이 `street`에 더 높은 attention을 줄 수 있다.

이처럼 Attention은 문맥에 따라 어떤 단어를 참고해야 하는지 동적으로 결정한다.

---

## 14. Attention의 한계도 있다

Attention이 강력하긴 하지만 한계도 있다.

### 14.1 계산량 문제

Self-Attention은 모든 토큰 쌍의 관계를 계산한다.

토큰 수가 n개라면 관계 계산은 대략 n²에 가까워진다.

```text
토큰 수가 1,000개 → 약 1,000 x 1,000 관계
토큰 수가 10,000개 → 약 10,000 x 10,000 관계
```

그래서 context window가 길어질수록 계산량과 메모리 사용량이 크게 증가한다.

이 문제를 줄이기 위해 FlashAttention, sparse attention, sliding window attention 같은 기법들이 등장했다.

### 14.2 Attention weight가 항상 설명은 아니다

Attention weight가 높다고 해서 반드시 모델의 최종 판단 이유라고 단정할 수는 없다.

모델 내부에는 Attention 외에도 Feed Forward Network, residual connection, layer normalization 등 여러 요소가 함께 작동한다.

따라서 Attention weight는 해석의 단서일 수는 있지만, 완전한 설명은 아니다.

---

## 15. 이번 편 핵심 정리

이번 편에서 가장 중요한 내용은 아래와 같다.

```text
1. 기존 Seq2Seq는 입력 전체를 하나의 context vector에 압축해야 했다.
2. 이 구조는 긴 문장에서 정보 손실 문제가 컸다.
3. Attention은 출력 시점마다 입력의 중요한 부분을 동적으로 참고한다.
4. Attention score는 현재 출력과 각 입력 위치의 관련도다.
5. Query, Key, Value는 검색 시스템처럼 이해하면 쉽다.
6. Query는 찾고 싶은 것, Key는 비교 대상의 색인, Value는 실제 가져올 정보다.
7. Self-Attention은 같은 문장 내부의 토큰들이 서로를 참고하는 구조다.
8. Transformer는 RNN을 제거하고 Self-Attention을 중심으로 문맥을 처리한다.
```

---

## 16. 꼭 기억해야 할 한 문장

> Attention은 모델이 모든 정보를 똑같이 보는 대신, 지금 필요한 정보에 더 집중하도록 만든 기술이다.

이 문장을 이해하면 Attention의 핵심은 잡은 것이다.

---

## 17. 다음 편 예고

다음 편에서는 드디어 Transformer를 다룬다.

다음 주제는 다음과 같다.

```text
LLM 공부 5편: Transformer 구조 이해
```

다음 편에서 다룰 내용은 아래와 같다.

```text
1. Transformer가 RNN을 제거한 이유
2. Encoder와 Decoder 구조
3. Self-Attention
4. Multi-Head Attention
5. Positional Encoding
6. Feed Forward Network
7. Residual Connection과 Layer Normalization
8. Transformer가 GPT와 BERT로 이어진 과정
```

Transformer를 이해하면 이후 GPT, BERT, LLM 학습 방식, RAG, Agent까지 훨씬 잘 연결된다.
