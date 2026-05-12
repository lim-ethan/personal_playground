# LLM 공부 8편: Tokenization과 Embedding

## 0. 이번 편의 목표

이번 편에서는 LLM이 사람의 문장을 어떻게 숫자로 바꾸고, 그 숫자를 어떻게 의미 공간에서 다루는지 살펴본다.

LLM은 텍스트를 그대로 이해하지 못한다.  
모든 입력 문장은 먼저 **토큰(Token)** 으로 쪼개지고, 각 토큰은 다시 **정수 ID** 로 바뀐다.  
그리고 이 정수 ID는 모델 내부에서 **벡터(Embedding)** 로 변환된다.

즉, LLM 입력의 실제 흐름은 다음과 같다.

```text
사람의 문장
→ Tokenization
→ Token ID
→ Token Embedding
→ Transformer Layer
→ 다음 토큰 예측
```

이번 편의 목표는 다음 질문에 답할 수 있게 되는 것이다.

```text
토큰이란 무엇인가?
왜 단어가 아니라 토큰 단위로 문장을 쪼갤까?
Token ID는 무엇인가?
Embedding은 무엇인가?
LLM 내부의 embedding과 RAG의 embedding은 같은 개념인가?
Context window는 tokenization과 어떤 관계가 있는가?
```

---

## 1. LLM은 텍스트를 직접 읽지 않는다

우리는 다음 문장을 자연스럽게 읽을 수 있다.

```text
오늘 서울의 날씨를 알려줘.
```

하지만 LLM 입장에서는 이 문장을 그대로 처리할 수 없다.  
딥러닝 모델은 기본적으로 숫자 연산을 수행하는 함수이기 때문이다.

그래서 텍스트를 숫자로 바꾸는 과정이 필요하다.

```text
"오늘 서울의 날씨를 알려줘."
→ [토큰들]
→ [토큰 ID들]
→ [벡터들]
```

이 과정의 첫 번째 단계가 **Tokenization**이다.

---

## 2. Tokenization이란?

Tokenization은 문장을 모델이 처리할 수 있는 작은 단위로 쪼개는 과정이다.

예를 들어 다음 문장을 보자.

```text
나는 보안 로그를 분석한다.
```

사람은 이 문장을 단어 단위로 볼 수 있다.

```text
나는 / 보안 / 로그를 / 분석한다
```

하지만 LLM의 tokenizer는 꼭 사람이 생각하는 단어 단위로 나누지 않는다.  
모델에 따라 다음처럼 나눌 수도 있다.

```text
나 / 는 / 보안 / 로그 / 를 / 분석 / 한다 / .
```

또는 영어에서는 이런 식으로 쪼개질 수 있다.

```text
unbelievable
→ un / believable
```

혹은

```text
tokenization
→ token / ization
```

이렇게 단어보다 작은 단위로 쪼개는 방식을 **Subword Tokenization**이라고 한다.

---

## 3. 왜 단어 단위로만 쪼개지 않을까?

처음 생각하면 단어 단위가 가장 자연스러워 보인다.

```text
I love security analysis
→ I / love / security / analysis
```

하지만 단어 단위 tokenization에는 큰 문제가 있다.

### 3.1 모르는 단어 문제가 생긴다

현실의 텍스트에는 계속 새로운 단어가 등장한다.

```text
신조어
제품명
사람 이름
도메인 이름
파일 경로
CVE 번호
해시값
IP 주소
```

보안 도메인에서는 특히 이런 문자열이 많다.

```text
CVE-2024-3094
malware_loader.exe
192.168.10.25
powershell_encoded_command
```

단어 단위 tokenizer를 쓰면 학습 때 보지 못한 단어를 처리하기 어렵다.

### 3.2 어휘 사전이 너무 커진다

모든 단어를 vocabulary에 넣으려면 사전 크기가 엄청나게 커진다.

```text
run
runs
running
runner
ran
```

이런 단어들을 모두 별도 항목으로 관리하면 비효율적이다.

### 3.3 여러 언어를 다루기 어렵다

LLM은 영어뿐 아니라 한국어, 일본어, 중국어, 코드, 수식, URL 등 다양한 형식을 처리해야 한다.

단어 기준은 언어마다 다르다.

```text
영어: 공백 기준 단어 구분이 비교적 명확함
한국어: 조사, 어미, 복합어 때문에 단어 경계가 복잡함
중국어/일본어: 공백 기준 단어 구분이 약함
코드: 자연어와 문법 구조가 다름
```

그래서 LLM은 보통 단어보다 유연한 **subword** 또는 **byte-level** 기반 tokenization을 사용한다.

---

## 4. 대표적인 Tokenization 방식

LLM에서 자주 등장하는 tokenization 방식은 다음과 같다.

```text
Word-level Tokenization
Character-level Tokenization
Subword Tokenization
Byte-level Tokenization
```

각각을 간단히 보자.

---

## 5. Word-level Tokenization

Word-level tokenization은 문장을 단어 단위로 나눈다.

```text
I love machine learning
→ I / love / machine / learning
```

장점은 직관적이라는 것이다.  
하지만 모르는 단어 문제와 너무 큰 vocabulary 문제가 있다.

예를 들어 학습 데이터에 `cybersecurity`라는 단어가 없었다면, 모델은 이 단어를 제대로 처리하지 못할 수 있다.

그래서 현대 LLM에서는 단어 단위만 사용하는 경우는 거의 없다.

---

## 6. Character-level Tokenization

Character-level tokenization은 문장을 문자 단위로 나눈다.

```text
security
→ s / e / c / u / r / i / t / y
```

장점은 모르는 단어 문제가 거의 없다는 것이다.  
어떤 단어든 문자로 쪼갤 수 있기 때문이다.

하지만 단점도 크다.

```text
문장이 너무 길어진다.
의미 단위를 학습하기 어렵다.
계산량이 증가한다.
```

예를 들어 `cybersecurity`라는 단어를 하나의 의미 있는 단위로 보지 못하고, 여러 문자 조합으로만 보게 된다.

그래서 character-level 방식은 너무 세밀하다.

---

## 7. Subword Tokenization

Subword tokenization은 단어보다 작지만 문자보다는 큰 단위로 문장을 쪼갠다.

예를 들어 다음과 같다.

```text
cybersecurity
→ cyber / security
```

또는

```text
unhappiness
→ un / happiness
```

이 방식은 단어 단위와 문자 단위의 중간 지점이다.

장점은 다음과 같다.

```text
모르는 단어를 조합해서 처리할 수 있다.
자주 나오는 단어는 하나의 토큰으로 유지할 수 있다.
드문 단어는 여러 subword로 쪼갤 수 있다.
사전 크기와 문장 길이 사이의 균형이 좋다.
```

현대 LLM의 tokenizer는 대부분 이 계열이다.

---

## 8. BPE란?

BPE는 Byte Pair Encoding의 약자다.

원래는 데이터 압축 알고리즘이었지만, NLP tokenization에도 널리 사용된다.

핵심 아이디어는 간단하다.

```text
자주 함께 등장하는 문자 또는 문자열 쌍을 하나의 토큰으로 합친다.
```

예를 들어 처음에는 문자를 하나씩 본다.

```text
l o w
l o w e r
n e w e s t
w i d e s t
```

여기서 `l`과 `o`가 자주 같이 나오면 `lo`로 합친다.  
그다음 `lo`와 `w`가 자주 나오면 `low`로 합친다.

이런 식으로 자주 등장하는 조합을 점점 vocabulary에 추가한다.

결과적으로 tokenizer는 다음과 같은 균형을 만든다.

```text
자주 나오는 표현 → 하나의 토큰
드문 표현 → 여러 토큰의 조합
```

---

## 9. 한국어 Tokenization이 어려운 이유

한국어는 영어보다 tokenization이 까다롭다.

예를 들어 다음 문장을 보자.

```text
취약점을 분석했습니다.
```

사람은 대략 이렇게 이해한다.

```text
취약점 / 을 / 분석 / 했 / 습니다
```

하지만 tokenizer에 따라 다음처럼 다르게 쪼개질 수 있다.

```text
취 / 약 / 점 / 을 / 분석 / 했 / 습니다
```

또는

```text
취약 / 점을 / 분석 / 했습니다
```

한국어는 조사와 어미가 붙기 때문에 단순 공백 기준으로 단어를 나누기 어렵다.

```text
분석했다
분석했습니다
분석하고
분석하려면
분석된
분석되는
```

같은 어근이라도 형태가 계속 바뀐다.

그래서 한국어를 잘 처리하려면 tokenizer의 품질이 중요하다.

---

## 10. Token ID란?

Tokenizer가 문장을 토큰으로 쪼개면, 각 토큰은 정수 ID로 바뀐다.

예를 들어 다음과 같은 vocabulary가 있다고 해보자.

```text
"나는" → 1204
"보안" → 8301
"로그" → 5412
"분석" → 9010
"한다" → 3321
"." → 13
```

그러면 문장은 다음처럼 변환된다.

```text
나는 보안 로그 분석한다.
→ [1204, 8301, 5412, 9010, 3321, 13]
```

LLM은 원문 문자열을 직접 보는 것이 아니라, 이 token ID sequence를 입력으로 받는다.

```text
input_ids = [1204, 8301, 5412, 9010, 3321, 13]
```

이 token ID들이 다시 embedding layer를 통과하면서 벡터로 변환된다.

---

## 11. Embedding이란?

Embedding은 토큰을 벡터로 표현한 것이다.

토큰 ID 자체는 단순한 번호일 뿐이다.

```text
"보안" → 8301
"로그" → 5412
```

8301이라는 숫자가 5412보다 크다고 해서 의미적으로 더 크거나 중요한 것은 아니다.  
그냥 사전에서 부여된 번호다.

그래서 모델은 token ID를 그대로 계산하지 않고, 각 ID를 고차원 벡터로 바꾼다.

```text
"보안" → [0.12, -0.03, 0.88, ..., 0.21]
"로그" → [0.09,  0.14, 0.75, ..., -0.10]
```

이 벡터가 바로 token embedding이다.

---

## 12. Embedding Layer의 역할

LLM 내부에는 embedding table이 있다.

간단히 말하면 다음과 같은 거대한 lookup table이다.

```text
Token ID 0     → vector
Token ID 1     → vector
Token ID 2     → vector
...
Token ID 8301  → vector
...
```

입력 token ID가 들어오면, embedding layer는 해당 ID에 대응하는 벡터를 가져온다.

```text
[1204, 8301, 5412]
→ [embedding(1204), embedding(8301), embedding(5412)]
```

그 결과 입력 문장은 벡터 시퀀스가 된다.

```text
문장 = 토큰 ID 시퀀스
모델 입력 = 벡터 시퀀스
```

---

## 13. Embedding 차원

Embedding 벡터는 보통 수백에서 수천 차원이다.

예를 들어 다음과 같을 수 있다.

```text
embedding dimension = 768
embedding dimension = 1024
embedding dimension = 4096
embedding dimension = 8192
```

모델이 클수록 embedding dimension도 커지는 경향이 있다.

예를 들어 하나의 토큰이 4096차원 벡터로 표현된다면, 문장 전체는 다음과 같은 형태가 된다.

```text
토큰 개수: 10개
embedding dimension: 4096
입력 shape: [10, 4096]
```

배치까지 포함하면 보통 다음과 같은 텐서 형태가 된다.

```text
[batch_size, sequence_length, hidden_size]
```

예를 들어:

```text
[1, 10, 4096]
```

---

## 14. 의미는 embedding 공간에서 표현된다

Embedding의 핵심은 의미적으로 비슷한 토큰이나 문장이 벡터 공간에서 가깝게 위치하도록 학습된다는 점이다.

예를 들어 다음 단어들은 의미적으로 가까울 수 있다.

```text
보안
취약점
공격
위협
침해
```

반대로 다음 단어들은 상대적으로 멀 수 있다.

```text
보안
바나나
축구
커피
```

물론 LLM 내부의 token embedding은 단순 단어 사전이 아니다.  
Transformer layer를 지나면서 문맥에 따라 계속 변환된다.

하지만 기본 아이디어는 이렇다.

```text
텍스트 의미를 벡터 공간의 위치로 표현한다.
```

---

## 15. Token Embedding과 Contextual Embedding

여기서 중요한 차이를 알아야 한다.

Embedding에는 크게 두 관점이 있다.

```text
Token Embedding
Contextual Embedding
```

### 15.1 Token Embedding

Token embedding은 embedding layer에서 나온 초기 벡터다.

같은 token ID라면 처음 embedding은 동일하다.

예를 들어 `bank`라는 토큰이 있다고 하자.

```text
river bank
bank account
```

초기 token embedding에서는 둘 다 같은 `bank` 벡터를 사용한다.

### 15.2 Contextual Embedding

하지만 Transformer layer를 지나면 문맥이 반영된다.

```text
river bank → 강둑 의미
bank account → 은행 의미
```

Self-Attention을 통해 주변 토큰들과 상호작용하면서 같은 토큰도 문맥에 따라 다른 표현이 된다.

이것이 contextual embedding이다.

즉, LLM 내부에서는 다음 흐름이 일어난다.

```text
초기 token embedding
→ Transformer layer 1
→ Transformer layer 2
→ ...
→ 문맥이 반영된 hidden state
```

---

## 16. LLM 내부 Embedding과 RAG Embedding은 같은가?

개념적으로는 비슷하지만, 목적이 다르다.

### 16.1 LLM 내부 Embedding

LLM 내부 embedding은 다음 토큰을 예측하기 위해 사용된다.

```text
목적: 생성 모델 내부 계산
입력: token ID
출력: token vector sequence
사용 위치: Transformer 내부
```

### 16.2 RAG Embedding

RAG에서 사용하는 embedding은 문서 검색을 위해 사용된다.

```text
목적: 의미 기반 검색
입력: 문장, 문단, chunk
출력: 하나의 문서/문장 벡터
사용 위치: vector DB, similarity search
```

예를 들어 RAG에서는 문서 chunk를 하나의 벡터로 만든다.

```text
"CVE-2024-3094는 XZ Utils 백도어 취약점이다."
→ [0.03, -0.22, 0.18, ..., 0.91]
```

그리고 사용자 질문도 벡터로 만든다.

```text
"XZ Utils 백도어 취약점 알려줘"
→ [0.05, -0.19, 0.20, ..., 0.88]
```

그다음 두 벡터의 유사도를 계산한다.

```text
cosine similarity
vector distance
inner product
```

정리하면 다음과 같다.

| 구분 | LLM 내부 Embedding | RAG Embedding |
|---|---|---|
| 목적 | 다음 토큰 예측 | 의미 기반 검색 |
| 단위 | 토큰 | 문장/문단/chunk |
| 출력 | 토큰별 벡터 시퀀스 | chunk 하나당 벡터 하나 |
| 사용 위치 | Transformer 내부 | Vector DB |
| 예시 | GPT embedding layer | bge, e5, text-embedding 계열 |

---

## 17. Context Window와 Token

LLM의 context window는 문자 수나 단어 수가 아니라 **토큰 수** 기준이다.

예를 들어 어떤 모델이 다음과 같은 context window를 가진다고 하자.

```text
context window = 8,192 tokens
```

이 말은 입력과 출력 전체를 합쳐 대략 8,192개 토큰 안에서 처리한다는 뜻이다.

```text
system prompt
+ user message
+ retrieved documents
+ conversation history
+ generated answer
≤ context window
```

RAG에서 chunk size를 정할 때 token 기준으로 잡아야 하는 이유가 여기에 있다.

---

## 18. 왜 RAG chunking에서 token 계산이 중요한가?

RAG에서는 문서를 chunk로 나눈 뒤, 검색된 chunk를 prompt에 넣는다.

예를 들어:

```text
사용자 질문: 100 tokens
시스템 프롬프트: 500 tokens
검색된 chunk 5개: 각 1,500 tokens
답변 생성 여유: 1,000 tokens
```

전체 token 수는 다음과 같다.

```text
100 + 500 + (1,500 × 5) + 1,000
= 9,100 tokens
```

만약 모델 context window가 8,192 tokens라면 초과된다.

그래서 RAG에서는 다음을 신경 써야 한다.

```text
chunk size
chunk overlap
검색 chunk 개수 top_k
system prompt 길이
conversation history 길이
답변 생성 max_tokens
```

너처럼 보안 문서 RAG를 만들 때는 이 부분이 특히 중요하다.  
정보보호 책, 취약점 설명, 위협 인텔리전스 문서는 문장이 길고 표, 목록, 코드, CVE 정보가 섞여 있기 때문이다.

---

## 19. Token 수는 언어마다 다르게 나온다

같은 의미라도 언어에 따라 token 수가 달라질 수 있다.

영어 문장:

```text
Explain the vulnerability analysis process.
```

한국어 문장:

```text
취약점 분석 절차를 설명해줘.
```

모델 tokenizer에 따라 한국어가 더 많은 토큰으로 쪼개질 수도 있고, 특정 모델에서는 비교적 효율적으로 처리될 수도 있다.

그래서 한국어 문서를 다룰 때는 반드시 실제 tokenizer로 token 수를 확인하는 습관이 필요하다.

---

## 20. Tokenization이 LLM 응답 품질에 주는 영향

Tokenizer는 단순 전처리 도구처럼 보이지만, 실제로 모델 성능에 영향을 준다.

특히 다음 영역에서 영향이 크다.

```text
한국어 처리 품질
코드 처리 품질
수식 처리 품질
URL, 경로, 로그 처리 품질
보안 IOC 처리 품질
긴 문서 처리 효율
```

예를 들어 보안 로그가 다음과 같다고 하자.

```text
powershell.exe -enc JABXAGMAPQBOAGUAdwAtAE8AYgBqAGUAYwB0AA==
```

이 문자열이 너무 잘게 쪼개지면 모델이 구조를 이해하기 어려울 수 있다.

또 다른 예시는 CVE다.

```text
CVE-2024-3094
```

이게 의미 있는 단위로 어느 정도 유지되는지, 아니면 너무 잘게 쪼개지는지는 보안 도메인에서 꽤 중요하다.

---

## 21. Special Token

LLM tokenizer에는 일반 단어 토큰 외에도 special token이 있다.

대표적으로 다음과 같은 것들이 있다.

```text
BOS: beginning of sequence
EOS: end of sequence
PAD: padding
UNK: unknown
SEP: separator
MASK: masked token
```

모델마다 special token 이름과 역할은 다를 수 있다.

예를 들어 대화형 모델에서는 사용자와 assistant의 발화를 구분하기 위한 토큰이 들어갈 수 있다.

```text
<|system|>
<|user|>
<|assistant|>
```

이런 special token은 chat template과 연결된다.

---

## 22. Chat Template과 Tokenization

ChatGPT류 모델은 단순히 사용자 문장만 입력받는 것이 아니다.

대화는 보통 내부적으로 다음과 같은 구조로 변환된다.

```text
system message
user message
assistant message
user message
assistant message
```

이 구조가 모델이 이해하는 특수 형식으로 바뀐다.

예를 들어 개념적으로는 다음과 비슷하다.

```text
<system>
너는 보안 분석 어시스턴트다.
</system>

<user>
최근 7일 incident 목록 보여줘.
</user>

<assistant>
```

이런 형식을 **chat template**이라고 볼 수 있다.

중요한 점은 chat template도 token을 사용한다는 것이다.

즉, 실제 context window에는 사용자 질문뿐 아니라 다음도 모두 포함된다.

```text
system prompt
role token
separator token
conversation history
retrieved context
tool call format
```

그래서 Agent나 RAG 시스템에서는 prompt가 생각보다 빨리 길어진다.

---

## 23. Tokenization과 비용

상용 LLM API에서는 보통 token 수에 따라 비용이 계산된다.

```text
입력 token 비용
출력 token 비용
```

그래서 tokenization은 비용과도 직접 연결된다.

예를 들어 다음 두 방식은 비용이 다르다.

```text
긴 문서 전체를 매번 prompt에 넣기
vs
필요한 chunk만 검색해서 넣기
```

RAG를 쓰는 이유 중 하나도 비용 효율성이다.

```text
전체 문서 입력 → 비싸고 느림
관련 chunk만 입력 → 저렴하고 빠름
```

---

## 24. Tokenization 실무 체크포인트

LLM 시스템을 만들 때는 다음을 확인하는 습관이 좋다.

```text
1. 내가 쓰는 모델의 tokenizer는 무엇인가?
2. 한국어 문서가 평균 몇 token으로 쪼개지는가?
3. chunk size는 문자 기준이 아니라 token 기준인가?
4. prompt 전체 token 수를 계산하고 있는가?
5. system prompt가 너무 길지는 않은가?
6. RAG top_k가 너무 크지는 않은가?
7. conversation history를 무한히 넣고 있지는 않은가?
8. 출력 max_tokens를 충분히 남겨두고 있는가?
```

특히 Agent 시스템에서는 tool 결과가 길어질 수 있다.

예를 들어 Cortex XDR에서 incident, case, asset, vulnerability 결과를 한꺼번에 가져오면 prompt가 금방 길어진다.

따라서 tool 결과도 요약하거나 필터링해서 넣어야 한다.

---

## 25. 간단한 예시: Tokenization 흐름

개념적으로 tokenizer는 다음 과정을 수행한다.

```text
입력 문장:
"최근 7일 high severity incident 보여줘"

토큰화:
["최근", "7", "일", "high", "severity", "incident", "보여", "줘"]

Token ID 변환:
[38120, 22, 918, 1947, 8721, 10923, 5521, 701]

Embedding 변환:
[vector_38120, vector_22, vector_918, ...]

Transformer 입력:
[sequence_length, hidden_size]
```

실제 tokenizer 결과는 모델마다 다르다.  
중요한 것은 텍스트가 반드시 token ID와 embedding을 거쳐 모델에 들어간다는 점이다.

---

## 26. RAG와 연결해서 이해하기

RAG 시스템에서는 tokenization과 embedding이 두 번 중요하다.

### 26.1 문서 저장 단계

```text
문서 로드
→ 문서 정제
→ chunk 분리
→ embedding 생성
→ vector DB 저장
```

여기서 chunk size를 token 기준으로 설계해야 한다.

### 26.2 질의 응답 단계

```text
사용자 질문
→ query embedding 생성
→ vector DB 유사도 검색
→ 관련 chunk 추출
→ prompt에 삽입
→ LLM 답변 생성
```

여기서 검색된 chunk들이 LLM context window를 초과하지 않도록 관리해야 한다.

즉, RAG에서는 다음 두 종류의 embedding을 구분해야 한다.

```text
검색용 embedding: vector DB 검색을 위한 embedding
생성 모델 내부 embedding: LLM이 token을 처리하기 위한 embedding
```

---

## 27. Agent와 연결해서 이해하기

Agent 시스템에서도 tokenization은 중요하다.

Agent prompt에는 보통 다음이 들어간다.

```text
system prompt
사용자 질문
사용 가능한 tool 목록
tool schema
이전 대화 기록
tool call 결과
중간 reasoning 또는 scratchpad 형식
최종 답변 지침
```

이 모든 것이 token을 사용한다.

그래서 tool이 많아질수록, schema가 길수록, 대화 기록이 길수록 context window를 많이 차지한다.

실무에서는 다음 전략이 필요하다.

```text
tool 설명을 짧고 명확하게 유지한다.
tool 결과를 필요한 필드만 남긴다.
긴 JSON 결과는 요약하거나 샘플링한다.
이전 대화 기록은 필요한 부분만 유지한다.
최종 답변에 필요한 정보만 prompt에 넣는다.
```

---

## 28. 이번 편 핵심 정리

이번 편의 핵심은 다음과 같다.

```text
LLM은 텍스트를 직접 이해하지 않는다.
텍스트는 token으로 쪼개진다.
token은 token ID로 바뀐다.
token ID는 embedding vector로 바뀐다.
embedding vector가 Transformer의 실제 입력이다.
context window는 token 수 기준이다.
RAG chunking도 token 기준으로 설계해야 한다.
LLM 내부 embedding과 RAG 검색 embedding은 목적이 다르다.
```

한 문장으로 요약하면 다음과 같다.

```text
Tokenization은 텍스트를 모델이 읽을 수 있는 단위로 바꾸는 과정이고,
Embedding은 그 토큰을 의미 있는 벡터 공간으로 옮기는 과정이다.
```

---

## 29. 스스로 점검할 질문

아래 질문에 답할 수 있으면 이번 편은 잘 이해한 것이다.

```text
1. LLM은 왜 텍스트를 바로 처리하지 못할까?
2. Tokenization이란 무엇인가?
3. Word-level tokenization의 한계는 무엇인가?
4. Subword tokenization은 왜 유용한가?
5. BPE의 핵심 아이디어는 무엇인가?
6. Token ID는 의미를 가진 숫자인가?
7. Embedding layer는 어떤 역할을 하는가?
8. Token embedding과 contextual embedding은 무엇이 다른가?
9. LLM 내부 embedding과 RAG embedding은 어떻게 다른가?
10. Context window가 token 기준이라는 말은 무슨 뜻인가?
11. RAG chunking에서 token 수를 왜 계산해야 하는가?
12. Agent 시스템에서 tool 결과가 길어지면 어떤 문제가 생기는가?
```

---

## 30. 다음 편 예고

다음 편에서는 **Pretraining과 Next Token Prediction**을 다룬다.

이번 편에서 텍스트가 token과 embedding으로 바뀌는 과정을 봤다면, 다음 편에서는 LLM이 그 token들을 이용해 어떻게 학습하는지 살펴본다.

다음 편의 핵심 질문은 다음과 같다.

```text
LLM은 정확히 무엇을 학습하는가?
Next Token Prediction이란 무엇인가?
왜 다음 토큰 예측만으로 언어 능력이 생기는가?
Pretraining 데이터는 어떤 역할을 하는가?
Base Model과 Instruct Model은 무엇이 다른가?
```
