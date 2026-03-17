# Future Goals: JSSP State Expansion (No Code Change Yet)

## 1) 목적
- Step-by-step 정책에서 상태 표현력을 높여 makespan 최소화 성능을 향상.
- Job index 토큰 편향에 덜 의존하고, 실제 상황(state) 기반 선택 강화.
- 추후 LoRA + RL(GRPO) 공통으로 사용할 수 있는 상태 설계 확립.

## 2) 현재 상태(기준)
현재 `state_json` 필드(10개):
1. `step_idx`
2. `total_steps`
3. `job_next_op`
4. `job_ready_time`
5. `machine_ready_time`
6. `next_machine`
7. `next_proc_time`
8. `remaining_ops`
9. `remaining_work`
10. `feasible_jobs`

## 3) 나중 목표: 추가 후보 상태
### A. Global
- `current_makespan`
- `progress_ratio` (`step_idx / total_steps`)
- `total_remaining_work`
- `bottleneck_machine_id`, `bottleneck_machine_load`

### B. Job-level
- `est_start_next_op[j]` (Earliest Start Time)
- `est_end_next_op[j]`
- `slack_like[j]` (간단한 urgency 지표)
- `job_completed_ratio[j]`

### C. Machine-level
- `machine_remaining_load[m]`
- `machine_utilization_like[m]` (진행 중 누적 기반 단순 지표)
- `queue_pressure[m]` (다음 op가 해당 machine인 job 수)

### D. Action(후보)별
- `estimated_makespan_after[action]`
- `delta_makespan[action]`
- `delta_bottleneck_load[action]`
- `next_feasible_count_after[action]`

## 4) 우선순위 (Phase Plan)
### Phase 1 (빠른 효과, 구현 난이도 낮음)
1. `current_makespan`
2. `est_start_next_op`
3. `estimated_makespan_after` / `delta_makespan`
4. `machine_remaining_load`

### Phase 2 (성능 고도화)
1. `bottleneck_machine_*`
2. `queue_pressure`
3. `slack_like`
4. `next_feasible_count_after`

### Phase 3 (대형 문제 일반화 집중)
1. 정규화/스케일링 규약 고정
2. 상태 압축 표현(토큰 길이 절감)
3. 후보 요약 템플릿 고정(학습/추론 동일)

## 5) 실험/검증 목표
- Feasibility: 100% 유지(마스킹 + env constraint 기준)
- 성능: TA/LA 기준 makespan 개선 여부 비교
- 안정성: 파싱 실패율 0, fallback 없음 유지
- 일반화: 작은 학습 크기 -> 큰 추론 크기에서 성능 저하율 측정

## 6) 적용 시 동시 수정 필요 범위(체크리스트)
- [ ] `llm_jssp/utils/jssp_step_env.py` 상태 필드 추가
- [ ] step 데이터 생성기(`generate_jssp_step_dataset.py`) 반영
- [ ] 프롬프트 빌더(`step_prompting.py` + 노트북 inline 코드) 반영
- [ ] 추론(`inference_jssp_fssp.py`, `colab_03`) 반영
- [ ] RL(`RL_jssp_fssp.py`, `colab_05`) 반영
- [ ] 스키마 버전 필드(`state_version`) 추가
- [ ] train/eval 데이터 재생성 및 회귀검증

## 7) 메모
- 현재 문서는 “나중 목표 정리”용이며 코드/실험 설정은 아직 변경하지 않음.
