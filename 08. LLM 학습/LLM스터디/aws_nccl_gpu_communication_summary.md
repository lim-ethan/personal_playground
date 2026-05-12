# AWS 환경에서 NCCL을 이용한 GPU 간 통신 정리

> 원문: AWS 기술 블로그  
> **「분산 트레이닝 관점에서의 AWS 인터커넥트 기술 소개 – AWS 환경에서 NCCL을 이용한 GPU 간 통신」**  
> 작성일: 2026-05-12  
> 작성자: Sangman Cho  
> URL: https://aws.amazon.com/ko/blogs/tech/nccl/

---

## 1. 문서 목적

이 문서는 AWS 환경에서 대규모 GPU 분산 학습을 수행할 때, GPU 간 통신이 어떤 방식으로 이루어지는지 이해하기 위한 정리 문서다.

특히 다음 요소들의 관계를 중심으로 설명한다.

- **NCCL**
- **EFA**
- **libfabric**
- **aws-ofi-nccl**
- **NVLink / NVSwitch**
- **GPUDirect RDMA**
- **AllReduce / AllGather / ReduceScatter**
- **Ring / Tree 알고리즘**
- **PyTorch DDP, FSDP, ZeRO-3**

핵심 메시지는 다음과 같다.

> AWS에서 멀티 노드 GPU 학습을 제대로 수행하려면  
> **PyTorch → NCCL → aws-ofi-nccl → libfabric → EFA → SRD**  
> 흐름을 이해해야 한다.

---

## 2. 왜 GPU 간 통신이 중요한가?

대규모 딥러닝 모델은 단일 GPU만으로 학습하기 어렵다.

LLM처럼 수십억~수백억 개 이상의 파라미터를 가진 모델은 여러 GPU에 작업을 나누어 학습한다. 이때 각 GPU는 서로 다른 데이터 배치를 처리하거나, 모델의 일부를 담당한다.

분산 학습에서는 보통 다음 과정이 반복된다.

1. 각 GPU가 서로 다른 데이터를 처리한다.
2. 각 GPU가 gradient를 계산한다.
3. 계산된 gradient를 다른 GPU들과 동기화한다.
4. 모든 GPU가 동일한 모델 업데이트를 수행한다.

이 과정에서 GPU 간 통신이 느리면 전체 학습 속도도 느려진다.

특히 데이터 병렬 학습에서는 매 학습 스텝마다 gradient 동기화가 필요하다. 수백 개 이상의 GPU를 사용하는 클러스터에서는 GPU 간 통신 병목이 전체 학습 시간을 크게 늘릴 수 있다.

---

## 3. MPI와 NCCL의 차이

분산 컴퓨팅에서는 CPU 중심 통신과 GPU 중심 통신을 구분해야 한다.

### 3.1 MPI

**MPI(Message Passing Interface)** 는 여러 노드에 분산된 CPU 프로세스들이 데이터를 주고받기 위한 표준 통신 인터페이스다.

대표 구현체는 다음과 같다.

- Open MPI
- Intel MPI

MPI는 주로 HPC 시뮬레이션, 과학 계산, CPU 기반 병렬 처리에서 사용된다.

### 3.2 NCCL

**NCCL(NVIDIA Collective Communications Library)** 은 NVIDIA가 만든 GPU 간 통신 라이브러리다.

NCCL은 다음을 지원한다.

- 단일 노드 내 GPU 간 통신
- 여러 노드 간 GPU 통신
- NVLink
- NVSwitch
- GPUDirect RDMA
- AllReduce
- AllGather
- ReduceScatter

즉, MPI가 CPU 프로세스 간 통신에 초점을 둔다면, NCCL은 GPU 간 통신에 특화되어 있다.

---

## 4. AWS EFA 환경의 소프트웨어 스택

AWS에서 멀티 노드 GPU 학습을 수행할 때는 보통 EFA를 사용한다.

EFA는 **Elastic Fabric Adapter** 의 약자로, AWS에서 고성능 노드 간 통신을 위해 제공하는 네트워크 인터페이스다.

NCCL이 AWS EFA를 사용하려면 다음 계층 구조를 거친다.

```text
Application
예: PyTorch, TensorFlow
        ↓
NVIDIA NCCL
        ↓
aws-ofi-nccl plugin
        ↓
libfabric
        ↓
EFA Provider
        ↓
EFA Hardware
        ↓
AWS Network Fabric
```

각 계층의 역할은 다음과 같다.

| 계층 | 역할 |
|---|---|
| Application | PyTorch, TensorFlow 같은 딥러닝 프레임워크 |
| NCCL | GPU 간 집합 통신을 최적화하고 실행 |
| aws-ofi-nccl | NCCL 요청을 libfabric API 호출로 변환 |
| libfabric | 고성능 네트워크 추상화 라이브러리 |
| EFA Provider | libfabric에서 EFA 하드웨어를 사용하기 위한 provider |
| EFA Hardware | AWS Nitro 기반 고성능 네트워크 어댑터 |
| SRD | AWS의 Scalable Reliable Datagram 프로토콜 |

중요한 점은 **NCCL이 libfabric을 직접 사용하지 않는다는 것**이다. AWS EFA 환경에서는 `aws-ofi-nccl` 플러그인이 NCCL과 libfabric 사이의 브리지 역할을 한다.

---

## 5. libfabric과 EFA Provider

**libfabric**은 고성능 네트워크 통신을 위한 표준 인터페이스를 제공하는 라이브러리다.

libfabric은 다양한 네트워크를 provider 형태로 지원한다.

예시는 다음과 같다.

- TCP
- InfiniBand
- EFA
- GNI

즉, 상위 애플리케이션은 네트워크 하드웨어의 세부 사항을 직접 알 필요 없이 libfabric API를 통해 고성능 통신을 수행할 수 있다.

EFA를 사용하려면 libfabric에 **EFA provider**가 포함되어 있어야 한다.

### Intel MPI 사용 시 주의점

Intel MPI는 자체 내장 libfabric을 사용할 수 있다. 그런데 Intel MPI의 내장 libfabric에는 EFA provider가 없을 수 있다.

이 경우 EFA 대신 TCP로 통신하게 되어 성능이 크게 떨어질 수 있다.

이를 방지하려면 다음 환경변수를 설정한다.

```bash
export I_MPI_OFI_LIBRARY_INTERNAL=0
export FI_PROVIDER=efa
```

의미는 다음과 같다.

| 환경변수 | 의미 |
|---|---|
| `I_MPI_OFI_LIBRARY_INTERNAL=0` | Intel MPI 내장 libfabric 사용 비활성화 |
| `FI_PROVIDER=efa` | EFA provider 명시 지정 |

---

## 6. NCCL의 집합 통신

분산 학습에서 GPU들이 서로 일대일로 계속 통신하면 매우 비효율적이다.

예를 들어 GPU가 64개라면 모든 GPU 쌍이 직접 연결되는 방식은 연결 수가 급격히 증가한다. 반면 Ring 같은 집합 통신 구조를 사용하면 각 GPU는 이웃 GPU와만 통신하면서도 전체 동기화를 수행할 수 있다.

NCCL은 이런 GPU 간 **집합 통신(Collective Communication)** 을 최적화한다.

대표적인 집합 통신 연산은 다음과 같다.

| 연산 | 의미 | 주요 사용처 |
|---|---|---|
| AllReduce | 여러 GPU의 값을 합산한 뒤 모든 GPU가 동일 결과를 가짐 | DDP gradient 동기화 |
| AllGather | 여러 GPU에 흩어진 값을 모아 모든 GPU가 전체 값을 가짐 | FSDP, ZeRO-3 파라미터 복원 |
| ReduceScatter | 합산한 결과를 다시 GPU별로 나누어 가짐 | FSDP, ZeRO-3 gradient shard 처리 |

---

## 7. AllReduce, AllGather, ReduceScatter 예시

3개의 GPU가 각각 gradient를 가지고 있다고 가정한다.

```text
GPU 0: [1, 2, 3]
GPU 1: [4, 5, 6]
GPU 2: [7, 8, 9]
```

### 7.1 AllReduce

AllReduce는 GPU별 값을 파라미터별로 합산하고, 모든 GPU가 동일한 결과를 갖게 한다.

```text
[1+4+7, 2+5+8, 3+6+9]
= [12, 15, 18]
```

결과:

```text
GPU 0: [12, 15, 18]
GPU 1: [12, 15, 18]
GPU 2: [12, 15, 18]
```

PyTorch DDP에서는 gradient 동기화에 주로 AllReduce를 사용한다.

### 7.2 AllGather

AllGather는 합산하지 않고 각 GPU의 값을 모두 모아서 모든 GPU가 동일하게 갖게 한다.

결과:

```text
GPU 0: [1, 2, 3, 4, 5, 6, 7, 8, 9]
GPU 1: [1, 2, 3, 4, 5, 6, 7, 8, 9]
GPU 2: [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

FSDP, ZeRO-3처럼 파라미터를 여러 GPU에 나누어 저장하는 방식에서, 특정 레이어 계산 시 필요한 파라미터 조각을 모을 때 사용된다.

### 7.3 ReduceScatter

ReduceScatter는 먼저 값을 합산한 뒤, 결과를 GPU별로 나누어 가진다.

합산 결과:

```text
[12, 15, 18]
```

GPU별 보유 결과:

```text
GPU 0: [12]
GPU 1: [15]
GPU 2: [18]
```

이 방식은 모든 GPU가 전체 결과를 복제하지 않기 때문에 메모리 사용량을 줄이는 데 유리하다.

---

## 8. NCCL의 자동 최적화

NCCL의 핵심 장점 중 하나는 하드웨어 토폴로지를 자동으로 감지하고 그에 맞게 통신을 최적화한다는 점이다.

NCCL은 초기화 시점에 다음 정보를 파악한다.

- GPU 간 NVLink 연결 구조
- NVSwitch 존재 여부
- PCIe 스위치 구조
- CPU socket / NUMA topology
- GPU와 NIC의 물리적 거리
- EFA 카드 위치

이를 바탕으로 NCCL은 다음을 자동으로 결정한다.

- Ring 또는 Tree 알고리즘 선택
- 병렬 channel 수
- 메시지 크기에 따른 프로토콜 선택
- GPU와 NIC 매핑
- NUMA 경계를 고려한 통신 경로

예를 들어 듀얼 소켓 서버에서 일부 GPU는 CPU 0 NUMA 노드에, 다른 GPU는 CPU 1 NUMA 노드에 연결되어 있을 수 있다. NCCL은 이런 구조를 감지해서 같은 NUMA 노드 내 통신을 먼저 수행하고, 느린 NUMA 간 통신은 최소화한다.

또한 GPU와 EFA NIC가 같은 PCIe 스위치에 연결되어 있는지도 고려해, GPU가 물리적으로 가까운 NIC를 통해 데이터를 전송하도록 매핑한다.

---

## 9. Ring과 Tree 알고리즘

NCCL은 데이터 크기와 GPU 개수에 따라 적절한 통신 알고리즘을 선택한다.

대표적인 알고리즘은 다음 두 가지다.

| 알고리즘 | 특징 | 적합한 상황 |
|---|---|---|
| Ring | GPU들이 원형으로 연결되어 데이터를 순차 전달 | 대용량 데이터 전송 |
| Tree | 계층 구조로 데이터를 모으고 분배 | 작은 데이터, 낮은 latency 필요 시 |

### 9.1 Ring 알고리즘

Ring은 GPU들이 원형으로 연결되어 인접 GPU와 데이터를 주고받는 방식이다.

장점:

- 네트워크 대역폭을 잘 활용한다.
- 대용량 데이터 전송에 유리하다.

단점:

- GPU 수가 늘어나면 latency가 증가할 수 있다.

### 9.2 Tree 알고리즘

Tree는 계층적으로 데이터를 모으는 방식이다.

장점:

- latency가 낮다.
- 작은 메시지 전송에 유리하다.

단점:

- 일부 단계에서는 모든 GPU가 동시에 일하지 않기 때문에 대역폭 활용도는 Ring보다 낮을 수 있다.

일반적으로 수 MB 이하의 작은 gradient에는 Tree가 유리하고, 수백 MB 이상의 대용량 파라미터 전송에는 Ring이 유리하다.

---

## 10. 파이프라이닝과 청킹

NCCL은 대용량 데이터를 한 번에 보내지 않는다.

대신 데이터를 작은 **chunk**로 나누고, 여러 chunk를 pipeline 방식으로 처리한다.

예를 들어 1GB 데이터를 전송한다고 하자.

청킹이 없다면:

```text
GPU 0이 1GB 전체 전송
→ GPU 1은 전체 데이터가 도착할 때까지 대기
→ 이후 연산 시작
```

청킹과 파이프라이닝을 사용하면:

```text
GPU 0이 첫 번째 chunk 전송
→ GPU 1은 첫 chunk를 받자마자 연산
→ GPU 0은 동시에 두 번째 chunk 전송
→ 여러 GPU 구간에서 전송과 연산이 동시에 진행
```

이렇게 하면 통신과 연산이 겹치면서 전체 시간이 줄어든다.

chunk가 너무 작으면 관리 오버헤드가 커지고, 너무 크면 pipeline이 충분히 채워지지 않는다. NCCL은 데이터 크기, channel 수, GPU 개수, 네트워크 지연시간을 고려해 chunk 크기를 자동으로 결정한다.

---

## 11. AWS에서 권장되는 NCCL 환경변수

최신 소프트웨어 스택에서는 대부분의 NCCL/EFA 관련 환경변수를 직접 튜닝하지 않는 것이 권장된다.

원문에서 언급한 기준은 다음과 같다.

- `libfabric v2.5.1 이상`
- `aws-ofi-nccl v1.19.1 이상`

현재 권장되는 최소 설정은 다음 정도다.

```bash
export FI_EFA_USE_HUGE_PAGE=0
```

이 설정은 PyTorch DataLoader에서 `num_workers > 0`을 사용할 때 fork 경로에서 huge page 할당 오류가 발생할 수 있는 문제를 회피하기 위한 것이다.

디버깅이 필요할 때는 다음을 사용할 수 있다.

```bash
export NCCL_DEBUG=INFO
```

다만 이 옵션은 로그가 많이 나오므로, 문제 진단 시에만 켜고 운영 환경에서는 끄는 것이 좋다.

### 튜닝 원칙

환경변수 튜닝은 처음부터 많이 하는 것보다, 실제 학습에서 병목이 확인된 뒤 제한적으로 진행하는 것이 좋다.

권장 흐름은 다음과 같다.

```text
1. 기본값으로 학습 실행
2. NCCL_DEBUG=INFO로 EFA 사용 여부 확인
3. 실제 병목 발생 시 nccl-tests로 기준 성능 측정
4. 환경변수를 한 번에 하나씩만 변경하며 비교
```

여러 환경변수를 동시에 바꾸면 어떤 설정이 효과를 냈는지 파악하기 어렵다.

---

## 12. 단일 노드 내 GPU 통신

단일 서버 내부에서는 GPU들이 보통 **NVLink**와 **NVSwitch**를 통해 통신한다.

AWS P5, P6 같은 고성능 GPU 인스턴스는 여러 GPU를 탑재하고 있으며, NVSwitch를 통해 GPU 간 고속 통신을 제공한다.

노드 내 GPU 통신 경로는 다음과 같다.

```text
GPU
→ NVLink
→ NVSwitch
→ NVLink
→ GPU
```

이 과정에서는 CPU나 시스템 메모리를 거치지 않는다.

원문 기준으로 설명된 대역폭은 다음과 같다.

| 인스턴스 계열 | GPU | 노드 내 GPU 통신 |
|---|---|---|
| P5 | H100, NVLink 4세대 | GPU당 양방향 900 GB/s |
| P6 | B200/B300, NVLink 5세대 | GPU당 양방향 1,800 GB/s |

노드 내 통신은 매우 빠르지만, 수백~수천 GPU를 사용하는 대규모 학습에서는 여러 노드 간 통신도 필요하다.

---

## 13. 멀티 노드 간 GPU 통신

여러 EC2 인스턴스에 걸쳐 GPU가 통신할 때는 EFA를 사용한다.

멀티 노드 GPU 통신 흐름은 다음과 같다.

```text
1. PyTorch DDP가 NCCL AllReduce 호출
2. NCCL이 알고리즘 선택 및 데이터 chunk 분할
3. aws-ofi-nccl이 NCCL 요청을 libfabric API로 변환
4. libfabric EFA provider가 EFA 하드웨어에 직접 접근
5. EFA가 SRD 프로토콜로 데이터를 네트워크 전송
6. 수신 측 EFA NIC가 GPUDirect RDMA로 GPU 메모리에 직접 기록
```

핵심은 CPU와 시스템 메모리를 최대한 우회하고, GPU 메모리와 EFA NIC 사이에서 직접 데이터를 주고받는 것이다.

### OS-bypass

EFA는 user space에서 하드웨어에 직접 접근할 수 있는 경로를 제공한다. 이를 통해 통신 과정에서 커널 개입을 줄이고 latency를 낮출 수 있다.

### GPUDirect RDMA

GPUDirect RDMA가 활성화되면 GPU 메모리에서 네트워크 카드로 데이터를 직접 전달할 수 있다.

일반적인 경로:

```text
GPU Memory
→ CPU / System Memory
→ NIC
```

GPUDirect RDMA 경로:

```text
GPU Memory
→ NIC
```

이렇게 CPU와 시스템 메모리 경유를 줄이면 통신 오버헤드를 줄일 수 있다.

---

## 14. 노드 내 통신과 노드 간 통신 비교

| 구분 | 노드 내 통신 | 노드 간 통신 |
|---|---|---|
| 주요 기술 | NVLink, NVSwitch | EFA, SRD, GPUDirect RDMA |
| 통신 경로 | 같은 서버 내부 GPU 간 직접 통신 | AWS 네트워크 패브릭 경유 |
| 지연시간 | 낮음 | 상대적으로 높음 |
| 대역폭 | 매우 높음 | 노드 내보다 낮음 |
| 예시 | 8 GPU 서버 내부 통신 | 여러 P5/P6 인스턴스 간 통신 |

원문에서는 p5.48xlarge의 경우 32개의 독립 네트워크 카드가 있고, 각 카드가 100 Gbps를 제공해 총 3,200 Gbps 대역폭을 제공한다고 설명한다.

하지만 노드 간 통신은 네트워크 스위치를 경유하므로, 노드 내 통신보다 대역폭과 latency 면에서 불리하다.

따라서 대규모 분산 학습에서는 노드 내 통신과 노드 간 통신의 차이를 이해하고, 통신 병목을 줄이는 것이 중요하다.

---

## 15. NCCL의 한계

NCCL은 AllReduce, AllGather, ReduceScatter처럼 대칭적인 집합 통신에 강하다.

그러나 최근 대규모 모델에서 많이 사용하는 **MoE(Mixture-of-Experts)** 구조는 통신 패턴이 더 복잡하다.

MoE에서는 입력 token마다 선택되는 expert가 달라진다.

이로 인해 다음과 같은 특징이 생긴다.

- GPU 간 전송량이 매 스텝마다 달라질 수 있다.
- 동적이고 비대칭적인 All-to-All 통신이 발생한다.
- 모든 GPU가 동일한 크기의 데이터를 주고받는다는 가정이 깨질 수 있다.

이런 패턴은 NCCL의 전통적인 집합 통신 모델만으로는 최적화가 어렵다.

원문에서는 다음 글에서 이 한계를 보완하는 기술을 다룬다고 설명한다.

- GPUDirect Async / IBGDA
- NVSHMEM
- PPLX-kernels

---

## 16. 실무 관점 핵심 정리

팀 내부 공유용으로 보면, 가장 중요한 내용은 다음과 같다.

1. **NCCL은 GPU 간 통신을 최적화하는 핵심 라이브러리다.**
2. **AWS에서 멀티 노드 GPU 학습을 할 때는 EFA가 중요하다.**
3. **NCCL이 EFA를 제대로 쓰려면 `aws-ofi-nccl` 플러그인이 필요하다.**
4. **`aws-ofi-nccl`은 NCCL 요청을 libfabric API 호출로 변환한다.**
5. **libfabric은 EFA provider를 통해 EFA 하드웨어를 사용한다.**
6. **EFA는 SRD와 OS-bypass를 통해 고성능 통신을 제공한다.**
7. **GPUDirect RDMA는 GPU 메모리와 NIC 간 직접 통신을 가능하게 한다.**
8. **NCCL은 Ring, Tree, chunking, pipelining, topology detection을 자동으로 수행한다.**
9. **최신 환경에서는 대부분의 NCCL 튜닝보다 기본값 사용이 우선이다.**
10. **MoE처럼 동적·비대칭 통신이 많은 모델은 NCCL만으로 한계가 있을 수 있다.**

---

## 17. 한 줄 요약

AWS에서 대규모 GPU 분산 학습을 수행할 때, NCCL은 GPU 간 집합 통신을 최적화하는 핵심 계층이며, EFA 환경에서는 `aws-ofi-nccl → libfabric → EFA` 구조를 통해 노드 간 고성능 통신을 수행한다.

---

## 18. 참고 링크

- AWS 기술 블로그 원문: https://aws.amazon.com/ko/blogs/tech/nccl/
- NVIDIA NCCL: https://developer.nvidia.com/nccl
- aws-ofi-nccl GitHub: https://github.com/aws/aws-ofi-nccl
- AWS EFA: https://aws.amazon.com/hpc/efa/
