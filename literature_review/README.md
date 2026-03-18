# LLM-for-CO Literature Audit

이 폴더는 `awesome-fm4co`의 `LLMs for Combinatorial Optimization` 섹션을 전수 점검하기 위한 보조 산출물이다.

## 파일

- `llm_for_co_catalog.json`
- `llm_for_co_catalog.csv`

각 행에는 다음이 포함된다.
- 논문 메타데이터: 날짜, 제목, 링크, 문제, venue, remark
- `abstract_status`, `abstract`
- `method_status`, `method_snippet`
- 접근 실패 시 `notes`

## 수집 범위

- source: `awesome-fm4co`의 `LLMs for Combinatorial Optimization`
- total papers: **157**

자동 수집 커버리지:
- abstract 수집 성공: **133 / 157**
- method snippet 수집 성공: **131 / 157**
- 둘 다 성공: **130 / 157**

## 실패 원인

실패의 대부분은 아래 때문이다.
- `openreview.net` 접근 제한 (`403 Forbidden`)
- 일부 publisher page의 PDF 미제공
- 일부 PDF 텍스트 추출 실패

현재 수집 스크립트는 핵심 scheduling 논문 중 아래 2편에 대해서는 arXiv 미러를 사용한다.
- `STARJOB: Dataset for LLM-Driven Job Shop Scheduling`
- `ACCORD: Autoregressive Constraint-satisfying Generation for COmbinatorial Optimization with Routing and Dynamic attention`

즉 이 카탈로그는 “전부 읽었다”를 주장하기 위한 파일이 아니라, **전수 점검을 재현 가능하게 만드는 근거 파일**이다.

## 재생성

```bash
python3 scripts/collect_llm_co_literature.py
```
