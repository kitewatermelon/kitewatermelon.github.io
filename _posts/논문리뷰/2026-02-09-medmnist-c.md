---
title: "[논문리뷰] MedMNIST-C: Comprehensive benchmark and improved classifier robustness by simulating realistic image corruptions"
last_modified_at: 2026-02-09
layout: single
categories:
  - 논문리뷰
  - Medical-AI
  - Dataset
tags:
  - Dataset
  - MICCAI
excerpt: "MedMNIST-C: Comprehensive benchmark and improved classifier robustness by simulating realistic image corruptions (MICCAI 2024 Workshop on Advancing Data Solutions in Medical Imaging AI)"
use_math: true
classes: wide
---
> Accepted at MICCAI Workshop on Advancing Data Solutions in Medical Imaging AI (ADSMI @ MICCAI 2024) [[Paper](https://arxiv.org/pdf/2406.17536)] [[Github]( https://github.com/francescodisalvo05/medmnistc-api )]   
>  Francesco Di Salvo, Sebastian Doerrich, and Christian Ledig   
>  Tue, 25 Jun 2024

## 1. Introduction
의료 인공지능 분야에서 DNN은 꾸준한 성장을 해왔다. 그럼에도 불구하고, 전통적인 neural network는 adversarial samples와 distribution shifts라는 한계에 직면했다. 이 문제들은 보통 imaging machine, post-processing, 환자 개인의 특성, acquisition protocols 등의 차이에 의해 발생한다.natural imaing domain에서는 ImageNet-C benchmark 등의 데이터셋을 통해 해결되어 왔지만, medical imaging domain에서는 dermatology, digital pathology, blood microscopy 등 각각의 modality에서만 단편적으로 해결하고 있었다. 또한 evaluation metric은 accuracy, average corruption error (CE), corruption error of confidence (CEC), normalized CE, relative normalized CE 등과 class imbalance 환경을 대비한 balanced accuracy를 활용한 방법으로 측정되어 왔다. 이로 인해 단일화된 metric을 사용하지 않아 의료 영상 기법의 다양성을 다룰 수 없었다.
model의 robustness를 강화하기 위해서 MixUp, CutMix, AugMix 등의 augmentation 기법이 사용되지만, 이 방법들은 모든 데이터셋에서 효과적이지 않았다. 최근 연구에 따르면 modal의 특성을 반영한 targeted augmentation 기법이 효과적이었으나, 이 역시 histpathology에서만 검증됐을 뿐 아직 의료 도메인에서 폭 넓은 검증이 이루어지지 않았다.   
본 논문은 targeted augmentation의 잠재력에 동기를 얻어 12개의 데이터셋과 9개의 imaing modality를 가진 MedMNIST+에서 MedMNIST-C 데이터셋을 도입한다. dataset-specific하게 디자인된 corruptions은 real-world artifacts와 possible distribution shifts를 반영한다. 저자들은 디자인된 corruptions을 targeted augmentations으로 사용한다. 이 augmentation 기법은 open API를 통해 학습에 사용할 수 있으며 기존의 방법보다 더 의미있는 결과를 보인다.

## 2. The MedMNIST-C dataset
### 2.1. Corruptions
아래 표에 요약 되어 있듯이 저자들은 digital, noise, blur, color, and task-specific의 5개의 카테고리들로 corruption을 분류하여 MedMNIST+ 데이터셋(test)에 적용했다. 이때 ImageNet-C을 따라 5개의 severity 단계로 지정했다. 

<center>
<img src="{{ '/assets/img/medmnist-c/tab1.webp' | relative_url }}" width="80%">
</center>
<br>

corruptuion을 도입한 결과는 다음과 같다.

<center>
<img src="{{ '/assets/img/medmnist-c/fig1.webp' | relative_url }}" width="80%">
</center>
<br>

PathMNIST & BloodMNIST는 현미경 기반 조직·혈액 슬라이드 이미지를 사용한다.
- 이미지 획득 과정에서 stain deposits, air bubbles 같은 물리적 오염물이 발생할 수 있다.
- 광학 초점 문제로 인한 defocus blur가 나타날 수 있다.
슬라이드 이동 또는 스캐너 진동으로 motion blur가 발생할 수 있다.
- 조명 조건 및 스캐너 설정 차이에 따라 brightness, contrast, saturation 변화가 생길 수 있다.

PneumoniaMNIST, ChestMNIST, OrganMNIST (A, C, S)는 X-ray 영상을 사용한다.
- X-ray 특성상 Gaussian, speckle, impulse, shot noise에 취약하다.
- photon 통계적 특성으로 인한 shot noise가 빈번하다.
- 촬영 조건 차이로 brightness 및 contrast variation이 발생할 수 있다.
- Gaussian blur 및 gamma correction과 같은 intensity 변형이 나타날 수 있다.

DermaMNIST는 dermatoscope를 사용한 피부 영상이다.
- noise artifacts (Gaussian, speckle, impulse, shot noise)가 발생할 수 있다.
- blurring effects (defocus, motion, zoom blur)가 나타날 수 있다.
- color-based artifacts (brightness, contrast 변화)가 발생할 수 있다.
- camera overlay에서 비롯된 문자(character) artifact가 포함될 수 있다.
- 장비 구조로 인해 black corners와 같은 task-specific artifact가 나타날 수 있다.

RetinaMNIST는 fundus photography 기반 영상이다.
- 조명 조건에 민감하여 brightness 및 contrast 변화가 발생할 수 있다.
- electronic/sensor irregularities로 Gaussian 및 speckle noise가 나타날 수 있다.
- Gaussian blur 및 defocus blur가 빈번하게 발생한다.

TissueMNIST는 고속 현미경(high-throughput microscopy) 기반 영상이다.
- impulse noise가 자주 발생한다.
- Gaussian blur가 나타날 수 있다.
- uneven illumination 문제가 존재하며, 이는 brightness 및 contrast 불균형으로 이어질 수 있다.

OCTMNIST는 Optical Coherence Tomography 기반 영상이다.
- speckle noise가 대표적 artifact로 나타난다.
- motion blur 및 defocus blur가 발생할 수 있다.
- speckle noise는 제한된 spatial-frequency bandwidth에서 기인하며 이미지 contrast를 저하시킨다.

BreastMNIST는 유방 초음파 영상이다.
- 초음파 산란으로 인한 speckle noise가 빈번하다.
- brightness variation 및 low contrast 문제가 발생할 수 있다.
- motion blur가 관찰될 수 있다.

### 2.2. Robustness measures

MedMNIST+는 binary classification, multiclass classification, multi-label (binary) classification, ordinal regression(multiclass classification으로 다룸)의 다양한 task를 포함하고 있다. evaluation의 일관성을 위해 ImageNet-C에서 영감을 받아 corruption error과 relative corruption error를 key metric으로 활용한다. 이때 class imbalance를 고려하려 1.0-balanced accuracy를 사용하여 balanced error를 측정한다. 

첫번째로 $BE_{clean}^{f}$을 MedMNIST+의 test셋에서 측정한다. 그 다음에 데이터셋 d에 specific한 corruption $c \in C_d$와 severity $s$에 대하여 $BE_{s,c}^f$ 를 측정한다. 전체 severity에 대하여 AlexNet의 error를 이용하여 normalize 하여 $BE_{c}^f$ 를 얻는다. 

$$
\begin{equation}
BE^f_c = \frac{\sum^5_{s=1}BE^f_{s,c}}{\sum^5_{s=1}BE^{AlexNet}_{s,c}}
\end{equation}
$$

또한 clean한 test set에서 얼마나 성능이 떨어졌는지 측정하기 위해 relative balanced error도 측정한다. 이는 모델의 robustness를 측정한다.

$$
\begin{equation}
rBE^f_c = \frac{\sum^5_{s=1}(BE^f_{s,c}-BE^f_{clean})}{\sum^5_{s=1}BE^{AlexNet}_{s,c}-BE^{AlexNet}_{clean}}
\end{equation}
$$

## 3 Experimental results
### 3.1. Robustness
이 섹션에서는 target-specific corruptions이 주로 사용되는 모델에서 얼마나 효과적인지 확인한다. AlexNet(baseline), ResNet50, DenseNet121, Vit-B/16, VGG16을 비교한다. 이때 모델은 ImageNet으로 pre-trained되었으며 각각의 MedMNIST+에서 학습한다. 학습 조건은 다음과 같다.

- epochs: 100
- Early Stopping
- optimizer: AdamW
- LR: 1e-4
- scheduler: cosine annealing 

robustness를 측정하기 위해 3개의 seed에서 BE와 rBE를 측정하여 평균을 낸 값을 보고한다. 
Vision transformers are robust learners에서 기대할 수 있듯이 가장 robust했다. 이 부분은 Digital과 Noise에서 가장 두드러지는 결과이다. 자세한 성능은 아래 표를 참고하면 된다.

<center>
<img src="{{ '/assets/img/medmnist-c/tab2.webp' | relative_url }}" width="80%">
</center>
<br>

### 3.2. Data augmentation
targeted augmentation의 효과를 확인하기 위해서 자주 사용되는 augmentation 기법과 targeted augmentation을 ResNet18에 대하여 비교한 결과이다. RandAugment은 targeted augmentation과 color, contrast, brightness, sharpness를 포함하기 때문에 single corruption으로 진행했으며 같은 논리로 AugMix는 RandAugment을 여러번 하는 것 이기 때문에 실험에서 제외했다. (저자는 In future work, the extension of our targeted augmentations along multiple chains could be explored.라고 언급한다.) 결과는 아래와 같다.

<center>
<img src="{{ '/assets/img/medmnist-c/tab2.webp' | relative_url }}" width="80%">
</center>
<br>

## 4. conclusion
본 연구에서는 넓은 범위의 medical imaging 스펙트럼에서 발생할 수 있는 이미지 손상의 맥락에서 알고리즘의 robustness를 평가하기 위한 벤치마크인 MedMNIST-C를 소개한다. 또한 의료 영상에 맞는 targeted augmentation을 적용하는 것의 이점을 보여준다. 마지막으로 재현 가능한 dataset과 API를 제공함으로 커뮤니티가 medical image의 견고성을 평가하고 향상하도록 장려한다. 

## 개인적인 생각
임상에서 꽤 유용한 데이터셋과 augmentation 기법인 것 같다. 제공하는 metric은 전체 severity를 평균내는 것이 약간 아쉽다. severity level 별로 robust 함을 입증해보도록 해야겠다.