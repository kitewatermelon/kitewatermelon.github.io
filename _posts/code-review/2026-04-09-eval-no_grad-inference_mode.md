---
title: "[코드 리뷰] eval() vs no_grad() vs inference_mode()"
last_modified_at: 2026-04-10
layout: single
categories:
  - Code-Review
tags:
  - PyTorch-basic
excerpt: "eval() vs no_grad() vs inference_mode()의 차이를 알아본다."
use_math: true
classes: wide
---

code 다운로드: [📥 eval-no_grad-inference_mode.ipynb 다운로드](assets/code/code-review/eval-no_grad-inference_mode.ipynb)

딥러닝 코드를 보다보면 model.eval(), with torch.no_grad() 그리고 with torch.inference_mode()를 많이 보곤한다. 오늘은 이 세 함수의 역할에 대하여 알아본다. 또한 with torch.inference_mode()가 왜 

## eval()
nn.Dropout()과 nn.BatchNormNd() 클래스는 대표적으로 train과 eval일때 역할이 다른 함수이다.

nn.Dropout의 [공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html)에 따르면
- train 시 입력의 일부 요소를 0으로 바꾸고 $\frac{1}{1-p}$로 나머지 값들을 scaling한다.
- eval 시에는 모든 입력을 그대로 사용한다.
- 이 방법을 사용하면 eval()시에 스케일링을 따로 하지 않아도 되어 불필요한 오버헤드를 줄이는 데 도와준다.

nn.BatchNormNd의 [공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)에 따르면
- train 시 var를 두번 계산하는데, forward 계산용: biased (N으로 나눔, correction=0)과 running_var 저장용: unbiased (N-1로 나눔, correction=1)
- eval 시에는 train때 쌓아둔 running mean/var로 정규화하여 사용한다.

위와 같이 train과 eval의 동작이 다른 모듈을 효과적으로 train과 eval로 제어하기 위해 .train()과 .eval()을 사용하게 된다. 아래 예제 코드가 도움이 되길 바란다.


```python
import torch
import torch.nn as nn

# ============================================================
# 1. model.eval() — BN/Dropout 동작 모드 전환 (gradient와 무관)
# ============================================================

print("=" * 60)
print("1. model.eval()은 BN/Dropout 모드만 바꾼다")
print("=" * 60)

# --- Dropout ---
dropout = nn.Dropout(0.8)
x = torch.ones(1, 8)

dropout.train()
print("train:", dropout(x))

dropout.eval()
print("eval: ", dropout(x))

# --- BatchNorm ---
bn = nn.BatchNorm1d(4)
for i in range(5):
    data = torch.randn(8, 4) + i
    bn(data)
    print(f"  step {i}: running_mean = {bn.running_mean.tolist()}")

bn.eval()
print("eval output:", bn(torch.zeros(2, 4)))

# --- eval이어도 gradient는 살아있다 ---
model = nn.Linear(4, 2)
model.eval()
inp = torch.randn(1, 4, requires_grad=True)
out = model(inp)
out.sum().backward()  # 정상 동작!
print("eval 모드에서 backward:", inp.grad is not None)  # True
```

    ============================================================
    1. model.eval()은 BN/Dropout 모드만 바꾼다
    ============================================================
    train: tensor([[5., 0., 0., 5., 0., 0., 5., 0.]])
    eval:  tensor([[1., 1., 1., 1., 1., 1., 1., 1.]])
      step 0: running_mean = [0.02741658128798008, 0.01935068890452385, -0.012505004182457924, 0.10418272018432617]
      step 1: running_mean = [0.10060923546552658, 0.13381418585777283, 0.021229349076747894, 0.23009954392910004]
      step 2: running_mean = [0.3041425943374634, 0.3139670193195343, 0.279056578874588, 0.3843126893043518]
      step 3: running_mean = [0.5296856760978699, 0.5720039010047913, 0.5548611283302307, 0.650330126285553]
      step 4: running_mean = [0.8712624907493591, 0.9221630096435547, 0.8863516449928284, 0.9963715672492981]
    eval output: tensor([[-0.9029, -0.9566, -0.9251, -1.0156],
            [-0.9029, -0.9566, -0.9251, -1.0156]],
           grad_fn=<NativeBatchNormBackward0>)
    eval 모드에서 backward: True


## torch.no_grad()
torch.no_grad()의 [공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html)에 따르면 no_grad()는 gradient calculation을 비활성화하는 Context-manager이다. 이를 통해 memory consumption을 줄이기 때문에 추론 단계에서 사용하는 것이 권장된다.

따라서 .eval()로 각 모듈을 eval 모드로 전환하고, with torch.no_grad() 안에서 추론을 함으로 계산 효율을 높혀준다.


```python
# ============================================================
# 2. torch.no_grad() — grad_fn 생략, version tracking은 유지
# ============================================================

print("=" * 60)
print("2. torch.no_grad()")
print("=" * 60)

x = torch.randn(3, requires_grad=True)

with torch.no_grad():
    a = x * 2

print("grad_fn:", a.grad_fn)           # None (autograd 그래프 안 만듦)
print("requires_grad:", a.requires_grad) # False
print("_version:", a._version)           # 0 (version tracking은 살아있음)

# 블록 밖에서 다시 autograd 텐서와 연산하면 grad 붙음
b = a + x
print("밖에서 a+x requires_grad:", b.requires_grad)  # True
```

    ============================================================
    2. torch.no_grad()
    ============================================================
    grad_fn: None
    requires_grad: False
    _version: 0
    밖에서 a+x requires_grad: True

## inference_mode() 
torch.inference_mode()는 no_grad()와 비슷한데, 추가적인 오버헤드를 줄여준다. [공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)에서 말하는 추가적인 오버헤드란 view tracking과 version counter bumps을 비활성화 하는 것이다.

[이 글도](https://docs.pytorch.org/serve/performance_checklist.html) 한번 쯤 읽는 것이 좋아보인다.


```python
# ============================================================
# 3. torch.inference_mode() — version tracking까지 제거
# ============================================================

print("=" * 60)
print("3. torch.inference_mode()")
print("=" * 60)

x = torch.randn(3, requires_grad=True)

with torch.inference_mode():
    c = x * 2

print("grad_fn:", c.grad_fn)             # None
print("requires_grad:", c.requires_grad)   # False
print("is_inference:", c.is_inference())   # True

try:
    print(c._version)
except RuntimeError as e:
    print(f"_version 접근 에러: {e}")
    # "Inference tensors do not track version counter."
```

    ============================================================
    3. torch.inference_mode()
    ============================================================
    grad_fn: None
    requires_grad: False
    is_inference: True
    _version 접근 에러: Inference tensors do not track version counter.


## version counter?
version counter의 역할은 다음과 같다. tensor의 내부 구현으로 tensor._version으로 read-only로 접근 가능하다. [참고](https://discuss.pytorch.org/t/how-to-get-the-version-numbers-of-a-modules-parameters/90726/2)


```python
# ============================================================
# 4. version counter가 왜 필요한지
# ============================================================

print("=" * 60)
print("4. version counter의 역할")
print("=" * 60)

x = torch.randn(3, requires_grad=True)
y = x * 2
z = y ** 2   # dz/dy = 2y → backward 때 y의 값이 필요
print("y._version:", y._version)  # 0

y.mul_(2)
print("y._version:", y._version)  # 1

try:
    z.backward(torch.ones(3))
except RuntimeError as e:
    print(f"backward 에러: {e}")
```

    ============================================================
    4. version counter의 역할
    ============================================================
    y._version: 0
    y._version: 1
    backward 에러: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [3]], which is output 0 of MulBackward0, is at version 1; expected version 0 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True, check_nan=False).

