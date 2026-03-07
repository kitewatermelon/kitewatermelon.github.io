# Kite's Blog

Medical AI & Computer Vision 연구 블로그

## Structure

```
_posts/
├── paper-review/          # 논문 리뷰
├── code-review/           # 코드 리뷰
└── study/
    └── medical/           # 의료 도메인 공부

assets/
├── img/{category}/{post-name}/
└── code/{category}/
```

## Categories

글의 **형식**에 따라 분류합니다.

| Category | Description | Path |
|----------|-------------|------|
| `Paper-Review` | 논문 리뷰 | `/categories/paper-review/` |
| `Code-Review` | 코드 구현/리뷰 | `/categories/code-review/` |
| `Study` | 도메인 공부/정리 | `/categories/study/` |

## Tags

글의 **내용**에 따라 분류합니다.

### Domain (도메인)
| Tag | Description |
|-----|-------------|
| `Medical-AI` | 의료 AI 관련 |
| `Computer-Vision` | 컴퓨터 비전 |

### Method (기법/주제)
| Tag | Description |
|-----|-------------|
| `Contrastive-Learning` | 대조 학습 |
| `Self-Supervised-Learning` | 자기 지도 학습 |
| `Foundation-Model` | 파운데이션 모델 |
| `Wavelet` | 웨이블릿 변환 |
| `Segmentation` | 분할 |
| `Information-Theory` | 정보 이론 |
| `Dataset` | 데이터셋 |

### Conference/Journal (학회/저널)
| Tag | Description |
|-----|-------------|
| `MICCAI` | Medical Image Computing and Computer Assisted Intervention |
| `ICLR` | International Conference on Learning Representations |
| `ICML` | International Conference on Machine Learning |
| `NIPS` | Neural Information Processing Systems |
| `ECCV` | European Conference on Computer Vision |

## Writing Guide

### Front Matter Template

```yaml
---
title: "[Paper Review] Your Paper Title"
last_modified_at: 2026-03-07
layout: single
categories:
  - Paper-Review          # 하나만 선택
tags:
  - Medical-AI            # 도메인
  - Contrastive-Learning  # 기법
  - MICCAI                # 학회
excerpt: "Brief description"
use_math: true
classes: wide
---
```

### Image Path Convention

```markdown
<img src="{{ '/assets/img/{category}/{post-name}/fig1.webp' | relative_url }}" width="80%">
```

예시:
- Paper Review: `/assets/img/paper-review/simclr/fig1.webp`
- Code Review: `/assets/img/code-review/contrastive-loss/fig1.webp`

## Local Development

```bash
bundle install
bundle exec jekyll serve
```

## License

Content: CC BY-NC-SA 4.0
Theme: [Minimal Mistakes](https://github.com/mmistakes/minimal-mistakes) (MIT)
