---
title: "[코드 리뷰] Dataloader의 동작과 역할"
last_modified_at: 2026-05-13
layout: single
categories:
  - Code-Review
tags:
  - PyTorch-basic
excerpt: "Dataloader의 동작과 역할을 알아본다."
use_math: true
classes: wide
---

<!-- code 다운로드: [📥 eval-no_grad-inference_mode.ipynb 다운로드](assets/code/code-review/eval-no_grad-inference_mode.ipynb) -->

PyTorch 라이브러리를 사용하여 딥러닝 코드를 작성하다보면 필수적으로 Dataset과 Dataloader 클래스를 만나게 된다. 보통의 경우 두 클래스의 정확한 동작과 역할을 잘 알지 못하며, 두 클래스의 역할을 혼용하기도 한다. 본 글을 통해 독자들이 Dataset과 Dataloader의 동작과 역할을 정확히 이해할 수 있길 바란다. 특히 본 글은 Dataloader 클래스에 초점을 둔 채로 작성되었으니, 이 부분 참고하여 읽으면 도움이 될 것 같다.

## Dataset과 Dataloader의 차이
Dataset과 Dataloader는 PyTorch 데이터 파이프라인의 핵심 두 축이지만, 각자의 역할은 명확히 구분된다.Dataset은 "무엇을 읽을 것인가"를 정의한다. 디스크에서 이미지를 읽거나, 레이블을 매핑하거나, 전처리(transform)를 적용하는 등 개별 샘플 하나를 어떻게 가져올지를 담당한다. __getitem__(index)를 구현하면 dataset[i]와 같이 특정 인덱스의 샘플을 꺼낼 수 있다.Dataloader는 "어떻게 꺼낼 것인가"를 정의한다. Dataset이 만들어둔 샘플을 어떤 순서로, 몇 개씩, 몇 개의 프로세스로 꺼낼지를 담당한다. 즉 배치 구성·셔플·병렬 로딩 등의 로직은 모두 Dataloader의 몫이다.
두 클래스를 혼용하는 흔한 실수는, 배치 처리나 셔플 로직을 Dataset 내부에 직접 구현하는 것이다. Dataset은 단일 샘플 반환에만 집중하고, 나머지는 Dataloader에 위임하는 것이 올바른 설계다.

이제 Dataloader에 대하여 알아보자. 

[torch.utils.data](https://docs.pytorch.org/docs/2.11/data.html#module-torch.utils.data)의 공식 문서는 다음과 같은 문장과 함께 시작된다. 
```
Eng: 
At the heart of PyTorch data loading utility is the torch.utils.data.DataLoader class. It represents a Python iterable over a dataset, with support for
- map-style and iterable-style datasets,
- customizing data loading order,
- automatic batching,
- single- and multi-process data loading,
- automatic memory pinning.

Kor:
PyTorch 데이터 로딩 유틸리티의 핵심은 torch.utils.data.DataLoader 클래스입니다. 이 클래스는 데이터셋을 순회할 수 있는 Python 이터러블로, 다음 기능들을 지원한다.
- map-style 및 iterable-style 데이터셋
- 데이터 로딩 순서 커스터마이징
- 자동 배치 처리
- 단일/다중 프로세스 데이터 로딩
- 자동 메모리 고정(pinning)
```

이 옵션들은 DataLoader의 생성자 인수를 통해 설정되며, 생성자의 시그니처는 다음과 같다:
```
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```

## map-style and iterable-style datasets
파이썬의 Iterable(이터러블)은 for 문이나 list(), tuple() 등에서 내부의 요소를 한 번에 하나씩 차례대로 반환(순회)할 수 있는 반복 가능한 객체를 말한다. 이들의 특징으로는 내부적으로 `__iter__()` 매서드나 `__getitem__()` 메서드를 구현하여 인덱싱이 가능한 객체라는 점이다.

Dataset 클래스는 데이터를 어떻게 읽을지 정의하는 클래스로 아래와 같이 두가지로 나뉜다.

- Map-style: `__getitem__(), __len__()` 인덱스로 접근하며 기본적인 Dataset 클래스가 이해 해당한다. 인덱스가 꼭 정수일 필요는 없으며, 문자열로도 사용 가능하다.
- Iterable-style: `__iter__()` 스트림 방식으로 접근하며, IterableDataset 라는 클래스로 정의한다. 흔히 사용되는 방식은 아니고, 랜덤 접근이 비싸거나 불가능한 경우 (DB 스트림, 원격 서버, 실시간 로그 등)에 사용한다.

보통 대부분의 데이터셋은 Map-style Dataset을 사용하므로, 해당 클래스에 집중하여 작성하겠다. Iterable-style Dataset을 다룬다면 공식문서를 읽어보는 것을 추천한다.

## customizing data loading order
이 섹션은 "데이터를 어떤 순서로 꺼낼 것인가" 를 결정하는 메커니즘을 설명한다. Map-style은 Sampler가 순서를 결정하게 되는데, shuffle=True로 지정을 하게 되면 내부적으로 RandomSampler를 만들어서 사용하게 된다. 

아래 표는 PyTorch에서 지원하는 Sampler 클래스들이니 필요한 곳에 사용하면 된다.

| Sampler 종류 | 동작 | 주요 파라미터 |
|---|---|---|
| `SequentialSampler` | 항상 같은 순서로 순차 샘플링 | `data_source` |
| `RandomSampler` | 랜덤 샘플링 (복원/비복원 선택) | `replacement`, `num_samples`, `generator` |
| `SubsetRandomSampler` | 주어진 인덱스 목록 내에서 랜덤 샘플링 (비복원) | `indices`, `generator` |
| `WeightedRandomSampler` | 가중치 확률 기반 샘플링 | `weights`, `num_samples`, `replacement` |
| `BatchSampler` | 다른 Sampler를 감싸서 배치 단위 인덱스 반환 | `sampler`, `batch_size`, `drop_last` |
| `DistributedSampler` | DDP 환경에서 프로세스별 데이터 구간 분할 | `num_replicas`, `rank`, `shuffle`, `seed`, `drop_last` |

## automatic batching
DataLoader는 `batch_size`, `drop_last`, `batch_sampler`, `collate_fn` 인수를 통해 개별로 가져온 데이터 샘플을 자동으로 배치로 묶는 기능을 지원한다.

### Automatic batching (default)
가장 일반적인 방식으로, 미니배치 단위로 데이터를 가져와 배치 샘플로 묶는다. 이때, `batch_size`와 `drop_last`는 본질적으로 sampler로부터 batch_sampler를 구성하는 데 사용된다. (`batch_size`와 `drop_last`는 사용자 편의를 위한 shortcut이고, 실제로는 batch_sampler로 변환되어 동작한다는 뜻)

```python
for indices in batch_sampler:
    yield collate_fn([dataset[i] for i in indices])
```
1. sampler가 인덱스를 하나씩 생성
1. batch_sampler가 그 인덱스를 batch_size개씩 묶음(drop_last도 여기서 처리)
1. dataset[i]로 각 샘플을 가져옴
1. collate_fn이 샘플 리스트를 하나의 배치 텐서로 변환

### collate_fn 다루기
Automatic batching이 활성화되면 `collate_fn`은 샘플 목록을 받아 배치로 묶어 반환한다. 기본 collate_fn(default_collate())의 동작은 다음과 같다:

1. 배치 차원 추가: 항상 새로운 첫 번째 차원을 배치 차원으로 추가
1. 자동 타입 변환: NumPy 배열, Python 수치값 → PyTorch 텐서로 자동 변환
1. 데이터 구조 보존: dict면 dict, list면 list, tuple이면 tuple 구조를 그대로 유지하되 값은 배치 텐서로

```python
# 개별 샘플 3개
[(img1, 0), (img2, 1), (img3, 0)]

# collate_fn 적용 후
(
    torch.stack([img1, img2, img3], dim=0),  # shape: (3, C, H, W)
    torch.tensor([0, 1, 0])           # shape: (3,)
)
```

Q1. 데이터셋 크기가 64개이고 batch_size=8일 때, DataLoader를 순회하면 몇 번 iterate되며, 각 iteration에서 반환되는 튜플의 구조는?

## single- and multi-process data loading
Python은 GIL 정책으로 인해 스레드간 완전한 병렬화가 불가능하다. 데이터 로딩이 연산 코드를 블로킹하는 것을 방지하기 위해, PyTorch는 `num_workers`를 양의 정수로 설정하는 것만으로 다중 프로세스 데이터 로딩으로 간단히 전환할 수 있다.

Map-style 데이터셋의 경우, 메인 프로세스가 sampler로 인덱스를 생성해 워커에 전달한다. 따라서 셔플 랜덤화는 메인 프로세스에서 처리되며, 워커는 할당받은 인덱스에 따라 데이터를 로딩하게 된다.
```
메인 프로세스
  └─ Sampler → [3, 1, 4, 2, 0, ...] 인덱스 생성
       └─ 워커 0 → dataset[3], dataset[1]
       └─ 워커 1 → dataset[4], dataset[2]
       └─ 워커 2 → dataset[0], ...
```
- 해당 작업은 `multiprocessing` 라이브러리에 의존하므로, 윈도우와 UNIX의 동작이 다르다. 
- 기본적으로 각 워커의 PyTorch 시드는 base_seed + worker_id로 설정된다.
- Q2. 왜 워커마다 서로 다른 난수를 사용할까?

## automatic memory pinning
Host → GPU 복사는 pinned(page-locked) 메모리에서 시작할 때 훨씬 빠르다. 

데이터 로딩 시, DataLoader에 `pin_memory=True`를 전달하면 가져온 데이터 텐서를 자동으로 pinned 메모리에 올려 CUDA GPU로의 데이터 전송 속도를 높일 수 있다.
기본 메모리 고정 로직은 텐서, 그리고 텐서를 담은 map/iterable만 인식한다.
텐서나 스토리지를 고정하면 비동기 GPU 복사도 사용할 수 있습니다. to()나 cuda() 호출 시 `non_blocking=True` 인수를 추가하면 되며, 이를 통해 데이터 전송과 연산을 오버랩시킬 수 있다.
DataLoader 생성자에 pin_memory=True를 전달하면 DataLoader가 pinned 메모리에 배치를 올려서 반환한다.

- Pinned memory는 RAM을 고정 점유하므로 메모리 부족 시 심각한 문제가 발생한다.
- Pinning 자체가 비싼 연산이라 데이터가 작거나 CPU 병목이 없으면 오히려 손해가 될 수 있다.
- 다중 프로세스 로딩에서 CUDA 텐서 직접 반환보다 `pin_memory=True` 사용이 권장된다.


오늘은 PyTorch Dataloader의 핵심 동작 방식을 살펴보았다. Dataset이 개별 샘플을 정의하는 역할이라면, Dataloader는 그 샘플을 어떤 순서로, 몇 개씩, 몇 개의 프로세스로 꺼낼지를 결정한다. Sampler로 순서를 제어하고, collate_fn으로 배치를 구성하며, num_workers와 pin_memory로 로딩 성능을 끌어올리는 구조를 이해하면 데이터 파이프라인 병목을 진단하고 최적화하는 데 큰 도움이 된다.

A1.
- iterate 횟수: 8번 (64 / 8)
- 각 iteration의 반환값:

```python
(
    tensor(...),  # shape: (8, C, H, W)  ← 이미지 배치
    tensor(...)   # shape: (8,)           ← 레이블 배치
)
```

튜플 1개 안에 배치 텐서 2개.

A2. 데이터 augmentation 시 다양성을 확보하기 위해, 같은 시드를 쓰면 모든 워커가 동일한 augmentation을 적용하게 되어 배치 내 다양성이 사라진다.