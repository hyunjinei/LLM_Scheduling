# solution_generation_english.py
from io import StringIO
import numpy as np
import re
import torch
import time
import traceback
import sys
from typing import List, Dict, Tuple, Any, Optional

try:
    from .jssp_masking_hooks import (
        build_logits_processors,
        build_prefix_allowed_tokens_fn_from_instance,
    )
except ImportError:  # pragma: no cover
    from jssp_masking_hooks import (
        build_logits_processors,
        build_prefix_allowed_tokens_fn_from_instance,
    )

def read_matrix_form_jssp(matrix_content: str, sep: str = ' '):
    """JSSP 문제 매트릭스 형식을 파싱합니다."""
    print(f"매트릭스 입력 파싱 시작: 길이 {len(matrix_content)} 문자")
    
    try:
        f = StringIO(matrix_content)
        first_line = next(f)
        
        n, m = map(int, first_line.split(sep))
        print(f"문제 크기: 작업 {n}개, 기계 {m}개")

        instance_lines = []
        for i in range(n):
            try:
                line = next(f).strip()
                instance_lines.append(line)
            except StopIteration:
                print(f"[ERROR] 줄 {i+1}을 읽는 중 예기치 않은 파일 끝에 도달")
                raise
        
        instance = np.array([line.split(sep) for line in instance_lines if line], dtype=np.int16)
        inst_for_ortools = instance.reshape((n, m, 2))

        try:
            ms_line = next(f).strip()
            ms = float(ms_line)
            print(f"최적 makespan: {ms}")
        except (StopIteration, ValueError) as e:
            ms = None
            print(f"최적 makespan 없음: {str(e)}")

        print(f"매트릭스 파싱 완료: {n}x{m} 문제")
        return n, m, inst_for_ortools.tolist(), ms
    
    except Exception as e:
        print(f"[ERROR] read_matrix_form_jssp 함수에서 예외 발생: {e}")
        print(traceback.format_exc())
        raise

def parse_solution(text):
    """
    솔루션 텍스트를 파싱하여 작업 및 makespan 정보를 추출합니다. (기존 함수)
    
    ⚠️ DEPRECATED: 이 함수는 더 이상 사용되지 않습니다.
    대신 parse_solution_order() + calculate_schedule() 조합을 사용하세요.
    
    이 함수는 구식 'JX-MY: start+duration -> end' 형식만 지원하며,
    현재 LLM이 출력하는 'Job X Operation Y, MZ' 형식을 처리할 수 없습니다.
    """
    print(f"⚠️ WARNING: parse_solution()은 deprecated 함수입니다. parse_solution_order() + calculate_schedule()을 사용하세요.")
    
    print(f"솔루션 텍스트 파싱 시작: 길이 {len(text)} 문자")
    
    # 형식 오류 추적
    format_errors = {
        "missing_job_lines": 0,
        "invalid_job_format": 0,
        "missing_operation_info": 0,
        "invalid_time_format": 0,
        "missing_makespan": 0,
        "total_errors": 0
    }
    
    # 기본 패턴 - Job 중심 포맷 (원본 형식) - 구식!
    pattern = r"Job\s*(\d+)\s*Operation\s*(\d+)\s*(?:on)?\s*Machine\s*(\d+)\s*:\s*(\d+)\s*\+\s*(\d+)\s*[-=]>\s*(\d+)"
    operations = re.findall(pattern, text, re.IGNORECASE)
    
    if operations:
        print(f"Job 중심 포맷으로 {len(operations)}개 작업 추출")
    else:
        # Machine 중심 포맷 시도
        machine_pattern = r"Machine\s+(\d+)\s+Consist:\s*"
        machine_blocks = re.split(machine_pattern, text)[1:]
        
        if len(machine_blocks) >= 2:  # 최소 Machine ID와 블록 하나 이상 필요
            operations = []
    
            for i in range(0, len(machine_blocks), 2):
                if i+1 < len(machine_blocks):
                    machine_id = machine_blocks[i].strip()
                    block_content = machine_blocks[i+1]
                    
                    # 블록 내 작업 추출
                    job_pattern = r"J(\d+)\s*Operation\s*(\d+):\s*(\d+)\s*\+\s*(\d+)\s*[-=]>\s*(\d+)"
                    job_matches = re.findall(job_pattern, block_content, re.IGNORECASE)
                    
                    for job, op, start, duration, end in job_matches:
                        operations.append((job, op, machine_id, start, duration, end))
            
            if operations:
                print(f"Machine 중심 포맷으로 {len(operations)}개 작업 추출")
        
        # 여전히 실패하면 추가 대체 패턴 시도
        if not operations:
            # 대체 패턴 - 일반 텍스트에서 작업 정보 추출
            job_line_pattern = r"(?:Job|J)\s*(\d+).*?(?:Operation|Op)\s*(\d+).*?(?:Machine|M)\s*(\d+).*?(\d+)\s*\+\s*(\d+)\s*[-=]>\s*(\d+)"
            operations = re.findall(job_line_pattern, text, re.IGNORECASE)
            
            if operations:
                print(f"대체 패턴으로 {len(operations)}개 작업 추출")
            else:
                format_errors["invalid_job_format"] = 1
                format_errors["total_errors"] += 1
                print("❌ 작업 정보 추출 실패 - 새 형식은 parse_solution_order()를 사용하세요")
    
    # Makespan 추출
    makespan_pattern = r"Makespan:\s*(\d+(\.\d+)?)"
    makespan_match = re.search(makespan_pattern, text, re.IGNORECASE)
    
    if makespan_match:
        makespan = float(makespan_match.group(1))
    else:
        # 대체 패턴들 시도
        alt_patterns = [
            r"makespan\s*[=:]\s*(\d+(\.\d+)?)",
            r"makespan\s+is\s+(\d+(\.\d+)?)",
            r"makespan.*?(\d+(\.\d+)?)",
        ]
        
        for pattern in alt_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                makespan = float(match.group(1))
                break
            else:
                makespan = None
                format_errors["missing_makespan"] = 1
                format_errors["total_errors"] += 1
    
    # 시간 형식 검증
    for job, operation, machine, start_time, duration, end_time in operations:
        try:
            start_int = int(start_time)
            duration_int = int(duration)
            end_int = int(end_time)
            if start_int + duration_int != end_int:
                format_errors["invalid_time_format"] += 1
                format_errors["total_errors"] += 1
        except ValueError:
            format_errors["invalid_time_format"] += 1
            format_errors["total_errors"] += 1
    
    # 결과 변환
    result = [
        {
            'Job': int(job),
            'Operation': int(operation),
            'Machine': int(machine),
            'Start Time': int(start_time),
            'Duration': int(duration),
            'End Time': int(end_time),
        }
        for job, operation, machine, start_time, duration, end_time in operations
    ]
    
    # Makespan이 0인 경우 처리
    if makespan is not None and makespan == 0:
        print(f"경고: Makespan이 0입니다. 유효하지 않은 값으로 처리합니다.")
        makespan = None
    
    print(f"파싱 완료: {len(result)}개 작업, makespan={makespan}, 형식 오류={format_errors['total_errors']}")
    return result, makespan, format_errors

def parse_solution_order(solution_text: str, is_debug_solution: bool = False) -> List[Dict[str, int]]:
    """
    LLM이 생성한 솔루션 텍스트에서 작업 순서만 추출합니다.
    'Job X Operation Y, MZ' 형식 및 기존 형식 모두 지원합니다.
    
    Args:
        solution_text (str): LLM이 생성한 솔루션 텍스트
        is_debug_solution (bool): 디버깅 출력 여부
        
    Returns:
        List[Dict[str, int]]: 각 연산의 Job, Operation, Machine 정보만 포함한 리스트
    """
    operations = []
    
    # 🧠 추론 모델 지원: </think> 태그 이후 부분만 파싱
    think_end_pos = solution_text.find("</think>")
    if think_end_pos != -1:
        # 추론 모델: </think> 이후 부분만 사용
        parsing_text = solution_text[think_end_pos + len("</think>"):]
        if is_debug_solution:
            print(f"🧠 추론 모델 감지: </think> 이후 {len(parsing_text)}자만 파싱")
    else:
        # 🔧 기존 로직: "assistant" 이후 부분 사용
        assistant_pos = solution_text.find("assistant")
        if assistant_pos != -1:
            # "assistant" 이후 부분만 사용
            parsing_text = solution_text[assistant_pos:]
            if is_debug_solution:
                print(f"🔧 기본 모델: assistant 이후 {len(parsing_text)}자만 파싱")
        else:
            # "assistant"가 없으면 전체 텍스트 사용
            parsing_text = solution_text
            if is_debug_solution:
                print(f"🔧 전체 텍스트 파싱: {len(parsing_text)}자")
    
    # 1. 새로운 'Job X Operation Y, MZ' 형식 시도
    new_format_pattern = r"Job\s*(\d+)\s*Operation\s*(\d+),\s*M(\d+)"
    new_format_matches = re.findall(new_format_pattern, parsing_text, re.IGNORECASE)
    
    if new_format_matches:
        print(f"새 형식 감지됨: {len(new_format_matches)}개 작업")
        
        # 🔍 디버깅: regex 매칭된 원본 텍스트들 확인 (첫 번째 솔루션만)
        '''
        디버깅: regex 매칭된 원본 텍스트들 확인 (첫 번째 솔루션만) 주석해제하고 실행
        '''
        # if is_debug_solution:
        #     print("\n" + "="*80)
        #     print("🔍 [DEBUG] Regex로 매칭된 원본 텍스트들:")
        #     print("="*80)
            
        #     # 매칭된 각 부분의 원본 텍스트 찾기 (parsing_text 기준)
        #     pattern_with_context = r"(Job\s*(\d+)\s*Operation\s*(\d+),\s*M(\d+))"
        #     context_matches = re.finditer(pattern_with_context, parsing_text, re.IGNORECASE)
            
        #     for i, match in enumerate(context_matches):
        #         matched_text = match.group(1)  # 전체 매칭된 텍스트
        #         job = match.group(2)
        #         operation = match.group(3)
        #         machine = match.group(4)
        #         start_pos = match.start()
        #         end_pos = match.end()
                
        #         # 앞뒤 20문자씩 컨텍스트 보기
        #         context_start = max(0, start_pos - 20)
        #         context_end = min(len(parsing_text), end_pos + 20)
        #         context = parsing_text[context_start:context_end]
                
        #         print(f"{i+1:2d}. 매칭: '{matched_text}' -> Job {job}, Op {operation}, M{machine}")
        #         print(f"    위치: {start_pos}-{end_pos}")
        #         print(f"    컨텍스트: '...{context}...'")
        #         print()
            
        #     print("="*80 + "\n")
        
        operations = [
            {
                'Job': int(job),
                'Operation': int(operation),
                'Machine': int(machine)
            }
            for job, operation, machine in new_format_matches
        ]
        print(f"새 형식으로 {len(operations)}개 작업 추출 완료")
        
        # # 🔍 디버깅: 추출된 작업들 전체 출력 (첫 번째 솔루션만)
        '''
        디버깅: regex 매칭된 원본 텍스트들 확인 (첫 번째 솔루션만) 주석해제하고 실행
        '''
        # if is_debug_solution:
        #     print("\n" + "="*60)
        #     print("🔍 [DEBUG] 추출된 작업들 전체 리스트:")
        #     print("="*60)
        #     for i, op in enumerate(operations):
        #         print(f"{i+1:2d}. Job {op['Job']}, Operation {op['Operation']}, Machine {op['Machine']}")
        #     print("="*60)
        #     print(f"🔍 [DEBUG] 총 추출된 작업 수: {len(operations)}개")
        #     print("="*60 + "\n")
        
        return operations
    
    # 2. Machine 중심 포맷 시도 (예: "Machine 0 Consist: \nJ1 Operation 0: 0 + 21 -> 21")
    machine_pattern = r"Machine\s+(\d+)\s+Consist:\s*"
    machine_blocks = re.split(machine_pattern, parsing_text)
    
    if len(machine_blocks) > 1:  # Machine 중심 포맷 감지
        print(f"Machine 중심 포맷 감지됨: {len(machine_blocks)-1}개 Machine 블록")
        
        # 첫 부분은 "Solution:" 등의 텍스트이므로 제외
        if not re.match(r'^\s*\d+\s*$', machine_blocks[0]):
            machine_blocks = machine_blocks[1:]
        
        # machine_id와 작업 내용을 순서대로 처리
        for i in range(0, len(machine_blocks), 2):
            if i+1 < len(machine_blocks):
                try:
                    machine_id = int(machine_blocks[i].strip())
                    block_content = machine_blocks[i+1]
                    
                    # 블록 내 작업 추출 - 시간 정보는 추출하지만 무시
                    job_pattern = r"J(\d+)\s*Operation\s*(\d+)"
                    job_matches = re.findall(job_pattern, block_content)
                    
                    # Machine 내 작업 순서대로 추가
                    for job, operation in job_matches:
                        operations.append({
                            'Job': int(job),
                            'Operation': int(operation),
                            'Machine': machine_id
                        })
                except ValueError:
                    print(f"Machine ID 파싱 실패: '{machine_blocks[i]}'")
        
        if operations:
            print(f"Machine 중심 포맷으로 {len(operations)}개 작업 추출 완료")
            return operations
    
    # 3. Job 중심 포맷 시도 (예: "Job 0 Operation 1 on Machine 2: 26 + 33 -> 59")
    job_pattern = r"Job\s*(\d+)\s*Operation\s*(\d+)\s*(?:on)?\s*Machine\s*(\d+)"
    job_matches = re.findall(job_pattern, parsing_text, re.IGNORECASE)
    
    if job_matches:
        print(f"Job 중심 포맷 감지됨: {len(job_matches)}개 작업")
        operations = [
            {
                'Job': int(job),
                'Operation': int(operation),
                'Machine': int(machine)
            }
            for job, operation, machine in job_matches
        ]
        print(f"Job 중심 포맷으로 {len(operations)}개 작업 추출 완료")
        return operations
    
    # 4. 대체 패턴 시도 (예: "J1 Op 0 M2: 59 + 17 -> 76")
    alt_pattern = r"(?:Job|J)\s*(\d+).*?(?:Operation|Op)\s*(\d+).*?(?:Machine|M)\s*(\d+)"
    alt_matches = re.findall(alt_pattern, parsing_text, re.IGNORECASE)
    
    if alt_matches:
        print(f"대체 패턴 감지됨: {len(alt_matches)}개 작업")
        operations = [
            {
                'Job': int(job),
                'Operation': int(operation),
                'Machine': int(machine)
            }
            for job, operation, machine in alt_matches
        ]
        print(f"대체 패턴으로 {len(operations)}개 작업 추출 완료")
        return operations
    
    print(f"작업 순서 추출 실패: 지원되는 포맷을 감지하지 못함")
    return []

def verify_problem_data(problem_data, operations_list, is_debug_solution: bool = False):
    """
    주어진 problem_data와 operations_list의 유효성을 검증합니다.
    필요한 경우 operations_list를 수정하여 유효한 operations 목록으로 변환합니다.
    
    Args:
        problem_data (List[List[List[int]]]): JSSP 문제 데이터 [job][operation] = [machine, duration]
        operations_list (List[Dict[str, int]]): 작업 순서 목록
        is_debug_solution (bool): 디버깅 출력 여부
        
    Returns:
        Tuple[List[Dict[str, int]], bool]: 수정된 operations 목록, 유효성 여부
    """
    if not problem_data:
        print("문제 데이터가 비어있습니다.")
        return operations_list, False
    
    if not operations_list:
        print("작업 목록이 비어있습니다.")
        return operations_list, False
    
    num_jobs = len(problem_data)
    if num_jobs == 0:
        print("작업이 없습니다.")
        return operations_list, False
    
    # 기계 수 확인 (첫 번째 작업의 작업 단계 수)
    num_machines = len(problem_data[0])
    if num_machines == 0:
        print("기계가 없습니다.")
        return operations_list, False
    
    print(f"문제 데이터 검증: {num_jobs}개 작업, {num_machines}개 기계")
    
    # 작업별 원래 머신 할당 맵 만들기
    machine_map = {}
    for job_id in range(num_jobs):
        machine_map[job_id] = {}
        for op_idx in range(min(num_machines, len(problem_data[job_id]))):
            try:
                machine, duration = problem_data[job_id][op_idx]
                machine_map[job_id][op_idx] = (machine, duration)
            except (IndexError, ValueError):
                if is_debug_solution:
                    print(f"문제 데이터 누락: 작업 {job_id}, 작업 단계 {op_idx}")
    
    # 수정된 operations 리스트 생성
    verified_operations = []
    operation_counts = {}  # 작업별 추가된 작업 단계 수 추적
    
    for op in operations_list:
        job_id = op['Job']
        op_idx = op['Operation']
        
        # 작업 ID 범위 확인
        if job_id < 0 or job_id >= num_jobs:
            if is_debug_solution:
                print(f"범위 초과 작업 ID 무시: {job_id} (유효 범위: 0-{num_jobs-1})")
            continue
        
        # 작업 단계 범위 확인
        if op_idx < 0 or op_idx >= num_machines:
            if is_debug_solution:
                print(f"범위 초과 작업 단계 무시: 작업 {job_id}, 작업 단계 {op_idx} (유효 범위: 0-{num_machines-1})")
            continue
        
        # 작업 단계가 주어진 문제 데이터에 있는지 확인
        if job_id not in machine_map or op_idx not in machine_map[job_id]:
            if is_debug_solution:
                print(f"작업 데이터 누락: 작업 {job_id}, 작업 단계 {op_idx}")
            continue
            
        # 정확한 기계 할당으로 수정
        correct_machine, duration = machine_map[job_id][op_idx]
        
        # 작업 단계 순서 추적 (각 작업이 순서대로 나타나는지 확인)
        if job_id not in operation_counts:
            operation_counts[job_id] = 0
        
        # 같은 작업의 이전 작업 단계가 모두 처리되었는지 확인
        if op_idx != operation_counts[job_id]:
            print(f"작업 순서 오류 무시: 작업 {job_id}, 작업 단계 {op_idx} (예상: {operation_counts[job_id]})")
            continue
        
        # 유효성 검사를 통과한 작업 추가 (정확한 기계 할당으로)
        verified_operations.append({
            'Job': job_id,
            'Operation': op_idx,
            'Machine': correct_machine  # LLM이 제안한 기계가 아닌 문제 데이터의 기계 사용
        })
        
        # 다음 작업 단계로 이동
        operation_counts[job_id] += 1
    
    # 모든 작업 단계가 포함되었는지 확인
    complete = True
    incomplete_jobs = []  # 불완전한 작업들의 상세 정보 저장
    
    for job_id in range(num_jobs):
        if job_id not in operation_counts or operation_counts[job_id] < num_machines:
            expected = num_machines
            actual = operation_counts.get(job_id, 0)
            incomplete_jobs.append(f"작업 {job_id}({actual}/{expected})")
            complete = False
    
    return verified_operations, complete, incomplete_jobs

def calculate_schedule(operations_list, problem_data, is_debug_solution: bool = False, save_schedule_file: str = None):
    """
    작업 순서와 문제 데이터를 사용하여 정확한 시작/종료 시간을 계산합니다.
    주어진 작업 순서를 정확히 따르며 JSSP 제약조건을 준수합니다.
    
    Args:
        operations_list (List[Dict[str, int]]): 작업 순서 (Job, Operation, Machine만 포함)
        problem_data (List[List[List[int]]]): 원래 JSSP 문제 데이터 [job][operation] = [machine, duration]
        is_debug_solution (bool): 디버깅 출력 여부
        save_schedule_file (str): 스케줄링 정보를 저장할 파일 경로 (None이면 저장하지 않음)
        
    Returns:
        Tuple[List[Dict[str, int]], int]: 시간이 계산된 스케줄, makespan
    """
    # 문제 데이터 검증 및 작업 순서 보정
    verified_operations, is_complete, incomplete_jobs = verify_problem_data(problem_data, operations_list, is_debug_solution=is_debug_solution)
    
    if not verified_operations:
        print("유효한 작업이 없어 스케줄을 계산할 수 없습니다.")
        return [], 0
    
    if not is_complete:
        if incomplete_jobs:
            print(f"❌ 불완전한 솔루션: {', '.join(incomplete_jobs)} - 스케줄링 skip")
        return [], 0  # 즉시 빈 스케줄 반환
    
    if is_debug_solution:
        print(f"작업 일정 재계산 시작: {len(verified_operations)}개 작업")
    
    # 스케줄링 정보 저장을 위한 리스트
    schedule_lines = []
    
    try:
        num_jobs = len(problem_data)
        num_machines = len(problem_data[0]) if num_jobs > 0 else 0
        
        # 각 작업의 완료된 작업 단계 수 추적
        completed_ops_count = {job_id: 0 for job_id in range(num_jobs)}
        
        # 각 작업의 이전 작업 단계 완료 시간
        job_completion_time = np.zeros(num_jobs)
        
        # 각 기계의 가용 시간
        machine_available_time = np.zeros(num_machines)
        
        # 최종 스케줄
        schedule = []
        
        # 작업 순서대로 스케줄링 (중요 변경: 동적 선택에서 정적 순서로 변경)
        for i, op_data in enumerate(verified_operations):
            job_id = op_data['Job']
            op_idx = op_data['Operation']
            machine_id = op_data['Machine']
            
            # 작업 시간 정보 얻기
            _, duration = problem_data[job_id][op_idx]
            
            # 시작 및 종료 시간 계산
            start_time = max(job_completion_time[job_id], machine_available_time[machine_id])
            end_time = start_time + duration
            
            # 스케줄에 추가
            schedule.append({
                'Job': job_id,
                'Operation': op_idx,
                'Machine': machine_id,
                'Start Time': int(start_time),
                'Duration': int(duration),
                'End Time': int(end_time)
            })
            
            # 상태 업데이트
            job_completion_time[job_id] = end_time
            machine_available_time[machine_id] = end_time
            completed_ops_count[job_id] += 1
            
            # 스케줄링 정보 출력 및 저장
            schedule_line = f"스케줄링: Job {job_id}, Operation {op_idx}, Machine {machine_id}, 시간 {start_time}-{end_time}"
            
            if is_debug_solution:
                print(schedule_line)
            
            # 파일 저장용으로 스케줄링 라인 저장
            schedule_lines.append(schedule_line)
        
        # 최종 makespan 계산
        makespan = max(op['End Time'] for op in schedule) if schedule else 0
        
        # 스케줄링 정보를 파일로 저장 (요청된 경우)
        if save_schedule_file and schedule_lines:
            try:
                import os
                os.makedirs(os.path.dirname(save_schedule_file), exist_ok=True)
                
                with open(save_schedule_file, 'w', encoding='utf-8') as f:
                    f.write(f"스케줄링 정보 (Makespan: {makespan})\n")
                    f.write("=" * 50 + "\n")
                    for line in schedule_lines:
                        f.write(line + "\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"최종 Makespan: {makespan}\n")
            except Exception as e:
                if is_debug_solution:
                    print(f"❌ 스케줄링 정보 저장 실패: {e}")
        
        return schedule, makespan
    
    except Exception as e:
        print(f"스케줄 계산 중 예외 발생: {e}")
        print(traceback.format_exc())
        return [], 0

def validate_schedule(schedule, problem_data, is_debug_solution: bool = False):
    """
    생성된 스케줄의 유효성을 검증합니다. makespan.py의 검증 로직을 직접 구현합니다.
    
    Args:
        schedule (List[Dict]): 검증할 스케줄
        problem_data (List): 원본 문제 데이터
        is_debug_solution (bool): 디버깅 출력 여부
        
    Returns:
        Tuple[bool, str]: (유효성 여부, 메시지)
    """
    if not schedule:
        return False, "빈 스케줄"
    
    if not problem_data:
        return False, "문제 데이터 없음"
    
    try:
        num_jobs = len(problem_data)
        num_operations_per_job = len(problem_data[0]) if num_jobs > 0 else 0
        total_expected_operations = num_jobs * num_operations_per_job
        
        # 🔥 새로운 완전성 검사 1: 총 operation 수 확인
        if len(schedule) < total_expected_operations:
            return False, f"불완전한 솔루션: {len(schedule)}/{total_expected_operations} 작업만 존재"
        
        # 작업 및 기계별 스케줄 구성
        jobs = {}
        machines = {}
        
        for op in schedule:
            job_id = op['Job']
            op_idx = op['Operation']
            machine_id = op['Machine']
            start_time = op['Start Time']
            duration = op['Duration']
            end_time = op['End Time']
            
            # 데이터 일관성 검사
            if end_time != start_time + duration:
                if is_debug_solution:
                    print(f"시간 계산 오류: 작업 {job_id}, 연산 {op_idx}, {start_time} + {duration} != {end_time}")
                return False, f"시간 계산 오류 (작업 {job_id}, 연산 {op_idx})"
            
            # 작업 및 기계별 그룹화
            if job_id not in jobs:
                jobs[job_id] = []
            jobs[job_id].append(op)
            
            if machine_id not in machines:
                machines[machine_id] = []
            machines[machine_id].append(op)
        
        # 🔥 새로운 완전성 검사 2: 모든 Job 존재 확인
        existing_jobs = set(jobs.keys())
        expected_jobs = set(range(num_jobs))
        missing_jobs = expected_jobs - existing_jobs
        if missing_jobs:
            return False, f"누락된 작업들: {sorted(missing_jobs)}"
        
        # 🔥 새로운 완전성 검사 3: 각 Job의 완전성 개별 확인
        for job_id in range(num_jobs):
            if job_id not in jobs:
                return False, f"작업 {job_id} 전체 누락"
            
            job_ops = jobs[job_id]
            expected_ops_count = len(problem_data[job_id])
            if len(job_ops) < expected_ops_count:
                return False, f"작업 {job_id} 불완전: {len(job_ops)}/{expected_ops_count} 작업 단계만 존재"
        
        # 1. 기계별 작업 겹침 검사
        for machine_id, ops in machines.items():
            sorted_ops = sorted(ops, key=lambda x: x['Start Time'])
            
            for i in range(len(sorted_ops) - 1):
                curr_op = sorted_ops[i]
                next_op = sorted_ops[i + 1]
                
                if curr_op['End Time'] > next_op['Start Time']:
                    # if is_debug_solution:
                    #     print(f"기계 {machine_id} 작업 겹침: {curr_op['Job']}/{curr_op['Operation']}(종료:{curr_op['End Time']}) → {next_op['Job']}/{next_op['Operation']}(시작:{next_op['Start Time']})")
                    return False, f"기계 {machine_id} 작업 겹침"
        
        # 2. 작업별 작업 단계 순서 검사 (수정됨: 문제 데이터 기준)
        for job_id, ops in jobs.items():
            sorted_ops = sorted(ops, key=lambda x: x['Operation'])
            
            # 🔥 수정: 문제 데이터 기준으로 완전성 확인
            op_indices = [op['Operation'] for op in sorted_ops]
            expected_indices = list(range(len(problem_data[job_id])))  # 문제에서 요구하는 전체
            
            if op_indices != expected_indices:
                if is_debug_solution:
                    print(f"작업 {job_id} 작업 단계 순서 오류: {op_indices} (예상: {expected_indices})")
                return False, f"작업 {job_id} 작업 단계 순서 오류: 누락 또는 중복 작업 단계"
            
            # 작업 단계 간 시간 중첩 검사
            for i in range(len(sorted_ops) - 1):
                curr_op = sorted_ops[i]
                next_op = sorted_ops[i + 1]
                
                if curr_op['End Time'] > next_op['Start Time']:
                    if is_debug_solution:
                        print(f"작업 {job_id} 순서 위반: 연산 {curr_op['Operation']}(종료:{curr_op['End Time']}) → 연산 {next_op['Operation']}(시작:{next_op['Start Time']})")
                    return False, f"작업 {job_id} 순서 위반"
        
        # 3. 문제 데이터와 기계 및 시간 일치 검사
        for job_id in jobs:
            for op in jobs[job_id]:
                op_idx = op['Operation']
                machine = op['Machine']
                duration = op['Duration']
                
                # 문제 데이터 범위 확인
                if job_id >= len(problem_data) or op_idx >= len(problem_data[job_id]):
                    if is_debug_solution:
                        print(f"문제 데이터 범위 벗어남: 작업 {job_id}, 연산 {op_idx}")
                    return False, "문제 데이터 범위 벗어남"
                
                expected_machine, expected_duration = problem_data[job_id][op_idx]
                
                if machine != expected_machine:
                    if is_debug_solution:
                        print(f"기계 불일치: 작업 {job_id}, 연산 {op_idx}, 할당 {machine} (예상: {expected_machine})")
                    return False, "기계 할당 불일치"
                
                if duration != expected_duration:
                    if is_debug_solution:
                        print(f"처리 시간 불일치: 작업 {job_id}, 연산 {op_idx}, 시간 {duration} (예상: {expected_duration})")
                    return False, "처리 시간 불일치"
        
        # 모든 검증 통과
        return True, f"완전하고 유효한 스케줄 ({len(schedule)}/{total_expected_operations} 작업)"
    
    except Exception as e:
        if is_debug_solution:
            print(f"스케줄 검증 중 예외 발생: {e}")
            print(traceback.format_exc())
        return False, f"검증 중 오류: {str(e)}"

def calculate_makespan(schedule: List[Dict[str, int]]) -> int:
    """
    스케줄에서 makespan을 계산합니다.
    
    Args:
        schedule (List[Dict[str, int]]): 완전한 스케줄
        
    Returns:
        int: 계산된 makespan
    """
    if not schedule:
        return 0
    
    makespan = max(op['End Time'] for op in schedule)
    print(f"계산된 makespan: {makespan}")
    return makespan

def format_solution(schedule: List[Dict[str, int]], makespan: int) -> str:
    """
    계산된 스케줄을 포맷에 맞추어 문자열로 변환합니다.
    새 형식: 'Job X Operation Y, MZ' 형식으로 출력하고 시간 계산 정보도 추가합니다.
    
    Args:
        schedule (List[Dict[str, int]]): 완전한 스케줄
        makespan (int): 계산된 makespan
        
    Returns:
        str: 포맷된 솔루션 문자열
    """
    # 시작 시간으로 정렬
    sorted_ops = sorted(schedule, key=lambda x: x['Start Time'])
    
    result_lines = ["Solution:"]
    result_lines.append("")  # 빈 줄 추가
    
    # 모든 작업을 시작 시간 순서대로 출력
    for op in sorted_ops:
        job_id = op['Job']
        op_idx = op['Operation']
        machine_id = op['Machine']
        start_time = op['Start Time']
        duration = op['Duration']
        end_time = op['End Time']
        
        # 'Job X Operation Y, MZ --> start + duration = end' 형식으로 출력
        result_lines.append(
            f"Job {job_id} Operation {op_idx}, M{machine_id} --> {start_time} + {duration} = {end_time}"
        )
    
    # Makespan 정보 추가
    result_lines.append("")  # 빈 줄 추가
    result_lines.append(f"Makespan: {makespan}")
    
    return "\n".join(result_lines)

def apply_chat_template_inference(prompt, tokenizer, index=0):
    """Applies chat template for inference. English prompts only."""
    # Prompt variations (English)
    user_variations = [
        "Instruction: Solve the following JSSP problem and provide the optimal job sequence. Also display the makespan.",
        "Task: Provide the solution for the following Job Shop Scheduling Problem as machine-wise job sequences.",
        "Command: Find a solution for the JSSP problem and present the machine-wise work order that minimizes makespan.",
        "Guide: Find the optimal schedule that can minimize the makespan for the following JSSP problem.",
    ]
    
    print(f"Applying prompt template: variation #{index}")
    
    # Basic constraints for JSSP problems (English)
    constraints = "Rules:\n" + "\n".join([
        "1. Each machine can process only one job at a time. (Operations on a machine cannot overlap)",
        "2. Jobs must follow the specified operation sequence. (e.g., Job 0's Operation 0 must complete before Job 0's Operation 1 starts)",
        "3. The goal is to minimize the total completion time (makespan).",
        "4. The next operation of the same job cannot start until the previous operation is completed.",
        "5. A machine cannot perform more than one operation simultaneously.",
        "6. Makespan is determined by the time when the last operation completes on all machines.",
        "7. Each job must be processed on the designated machine and cannot be changed to another machine.",
        "8. Jobs must follow the specified operation sequence. (Job 0: Operation 0 → Operation 1 → Operation 2...)",
        "9. Machines can have idle time between operations. Minimizing idle time can be an optimization goal.",
        "10. Each machine's operation must complete before starting the next operation.",
        "11. Even if multiple machines are available, one job cannot be processed in parallel on multiple machines.",
        "12. The goal is to minimize the total time (makespan) to complete all operations.",
        "13. A job is complete only after its last operation is processed.",
        "14. Each machine has a fixed processing capacity that cannot be exceeded.",
        "15. Jobs are processed in the order assigned to machines."
    ])
    
    # Add operation format explanation
    operation_format_explanation = "\n\nOperation Format Explanation:\n" + "\n".join([
        "In the 'Operation X: MY,Z' format:",
        "- X is the operation number (starting from 0)",
        "- Y is the machine number where this operation should be performed",
        "- Z is the processing time for this operation",
        "For example, 'Operation 0: M6,69' means the first operation is performed on machine 6 and takes 69 time units."
    ])

    # Updated work sequence selection strategy with strict constraints
    selection_strategy = "\n\nWork Sequence Selection Strategy:\n" + "\n".join([
        "- Ensure that operations of each job are executed in sequence, while finding an order that minimizes overall makespan.",
        "- Output in 'Job X Operation Y, MZ' format, listing each operation in execution order. For example: 'Job 0 Operation 0, M0' means Job 0's first operation is executed on Machine 0.",
        "- All operations from all jobs must be fully sequenced with no duplication or omission.",
        "- The total number of operations must equal (number of jobs) × (number of machines).",
        "- Double-check that each job's operations appear exactly once and in correct order."
    ])

    # Compose messages
    system_message = f"You are an expert in Job Shop Scheduling Problem (JSSP). Follow these rules:\n\n{constraints}\n{operation_format_explanation}\n{selection_strategy}"

    # Add validation checklist to user message
    user_message = (
        f"{user_variations[index]}\n{prompt}\n\n"
        "Check your answer against the following rules:\n"
        "- Each operation must appear exactly once.\n"
        "- No operation should be omitted.\n"
        "- The total number of operations must match (number of jobs × number of machines).\n"
        "- Maintain the correct order of operations for each job.\n"
        "- Ensure no machine is assigned overlapping operations.\n"
        "- Clearly state the makespan at the end."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    # Apply chat template
    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    return formatted_text

def gen(
    model,
    prompt,
    tokenizer,
    dev_map,
    maxlen=8192,
    sample=True,
    num_return_sequences=1,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    prefix_allowed_tokens_fn=None,
    logits_processor=None,
):
    """모델을 사용하여 텍스트를 생성합니다."""
    toks = tokenizer(prompt, return_tensors="pt")
    
    # dev_map이 "auto"인 경우 실제 장치로 변환
    actual_device = dev_map
    if dev_map == "auto":
        if torch.cuda.is_available():
            actual_device = "cuda:0"  # 첫 번째 CUDA 장치 사용
        else:
            actual_device = "cpu"     # CUDA 사용 불가능한 경우 CPU 사용
    
    with torch.no_grad():
        res = model.generate(
            **toks.to(actual_device),
            max_new_tokens=maxlen,
            do_sample=sample,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        ).to('cpu')
    return tokenizer.batch_decode(res, skip_special_tokens=True)

def analyze_machine_utilization(schedule):
    """
    각 기계의 활용도를 분석합니다.
    
    Args:
        schedule (List[Dict]): 계산된 스케줄
        
    Returns:
        Dict: 기계별 활용도 및 유휴 시간 정보
    """
    if not schedule:
        return {}
    
    # 기계별 작업 그룹화
    machines = {}
    makespan = max(op['End Time'] for op in schedule)
    
    for op in schedule:
        machine_id = op['Machine']
        if machine_id not in machines:
            machines[machine_id] = []
        machines[machine_id].append(op)
    
    # 기계별 분석
    machine_stats = {}
    for machine_id, ops in machines.items():
        # 시작 시간 기준으로 정렬
        sorted_ops = sorted(ops, key=lambda x: x['Start Time'])
        
        # 총 작업 시간 계산
        total_working_time = sum(op['Duration'] for op in sorted_ops)
        
        # 기계 유휴 시간 계산
        idle_time = 0
        last_end_time = 0
        
        for op in sorted_ops:
            if op['Start Time'] > last_end_time:
                idle_time += op['Start Time'] - last_end_time
            last_end_time = op['End Time']
        
        # 마지막 작업 이후 유휴 시간
        if last_end_time < makespan:
            idle_time += makespan - last_end_time
        
        # 활용률 계산
        utilization = (total_working_time / makespan) * 100
        
        machine_stats[machine_id] = {
            'utilization': utilization,
            'idle_time': idle_time,
            'total_working_time': total_working_time,
            'operations_count': len(sorted_ops)
        }
    
    return machine_stats

def analyze_waiting_times(schedule):
    """
    각 작업의 대기 시간을 분석합니다.
    
    Args:
        schedule (List[Dict]): 계산된 스케줄
        
    Returns:
        Dict: 작업별 총 대기 시간
    """
    if not schedule:
        return {}
    
    # 작업별 작업 단계 그룹화
    jobs = {}
    
    for op in schedule:
        job_id = op['Job']
        if job_id not in jobs:
            jobs[job_id] = []
        jobs[job_id].append(op)
    
    # 작업별 대기 시간 계산
    wait_times = {}
    
    for job_id, ops in jobs.items():
        # 작업 단계 순서대로 정렬
        sorted_ops = sorted(ops, key=lambda x: x['Operation'])
        
        # 총 대기 시간 계산
        total_wait_time = 0
        last_end_time = 0
        
        for op in sorted_ops:
            # 첫 번째 작업 단계가 아니라면 대기 시간 계산
            if last_end_time > 0 and op['Start Time'] > last_end_time:
                total_wait_time += op['Start Time'] - last_end_time
            
            last_end_time = op['End Time']
        
        wait_times[job_id] = total_wait_time
    
    return wait_times

def find_critical_path(schedule):
    """
    스케줄에서 병목 구간(임계 경로)을 찾습니다.
    
    Args:
        schedule (List[Dict]): 계산된 스케줄
        
    Returns:
        List[Dict]: 임계 경로에 있는 작업들
    """
    if not schedule:
        return []
    
    # makespan 구하기
    makespan = max(op['End Time'] for op in schedule)
    
    # 역방향으로 임계 경로 찾기
    critical_path = []
    current_time = makespan
    
    # 마지막으로 끝나는 작업 찾기
    last_op = max(schedule, key=lambda x: x['End Time'])
    critical_path.append(last_op)
    
    # 작업과 기계 제약 고려하여 역방향 추적
    while current_time > 0:
        # 현재 작업 가져오기
        current_op = critical_path[-1]
        current_job = current_op['Job']
        current_machine = current_op['Machine']
        current_start = current_op['Start Time']
        
        # 1. 같은 작업의 이전 작업 단계 찾기
        prev_op_same_job = None
        for op in schedule:
            if op['Job'] == current_job and op['Operation'] == current_op['Operation'] - 1:
                prev_op_same_job = op
                break
        
        # 2. 같은 기계의 이전 작업 찾기
        prev_op_same_machine = None
        for op in schedule:
            if op['Machine'] == current_machine and op['End Time'] == current_start:
                prev_op_same_machine = op
                break
        
        # 이전 작업 결정 (더 늦게 끝나는 작업이 임계 경로)
        prev_op = None
        if prev_op_same_job and prev_op_same_machine:
            prev_op = prev_op_same_job if prev_op_same_job['End Time'] > prev_op_same_machine['End Time'] else prev_op_same_machine
        elif prev_op_same_job:
            prev_op = prev_op_same_job
        elif prev_op_same_machine:
            prev_op = prev_op_same_machine
        
        # 이전 작업이 없으면 종료
        if not prev_op or prev_op['End Time'] == 0:
            break
        
        # 임계 경로에 추가
        critical_path.append(prev_op)
        current_time = prev_op['Start Time']
    
    # 시작 시간 순으로 정렬
    critical_path.sort(key=lambda x: x['Start Time'])
    
    return critical_path

def create_improvement_prompt(
    solution_text,
    calculated_makespan,
    jssp_problem,
    schedule=None,
    problem_data=None,
):
    """
    Build a stricter reflection prompt for improving one-shot JSSP schedules.
    """
    clean_solution = []
    for line in solution_text.split("\n"):
        if "Job" in line and "Operation" in line and "," in line:
            job_op_part = line.split("-->")[0].strip() if "-->" in line else line
            job_op_part = job_op_part.split("=")[0].strip() if "=" in line else job_op_part
            clean_solution.append(job_op_part)
    solution_lines = "\n".join(clean_solution)

    expected_ops = None
    if problem_data:
        try:
            expected_ops = sum(len(job_ops) for job_ops in problem_data)
        except Exception:
            expected_ops = None

    analysis_text = ""
    if schedule and len(schedule) > 0:
        machine_stats = analyze_machine_utilization(schedule)
        machine_analysis = "## Machine Utilization Analysis ##\n"
        for machine_id, stats in sorted(machine_stats.items()):
            machine_analysis += (
                f"- Machine {machine_id}: utilization {stats['utilization']:.1f}%, "
                f"idle {stats['idle_time']}, ops {stats['operations_count']}\n"
            )

        wait_times = analyze_waiting_times(schedule)
        wait_analysis = "\n## Job Waiting Time Analysis ##\n"
        high_wait_jobs = []
        for job_id, wait_time in sorted(wait_times.items()):
            if wait_time > 0:
                wait_analysis += f"- Job {job_id}: waiting {wait_time}\n"
                high_wait_jobs.append((job_id, wait_time))
        high_wait_jobs.sort(key=lambda x: x[1], reverse=True)

        critical_path = find_critical_path(schedule)
        bottleneck_analysis = "\n## Critical Path ##\n"
        for op in critical_path:
            bottleneck_analysis += (
                f"- Job {op['Job']} Op {op['Operation']} on M{op['Machine']}: "
                f"{op['Start Time']}->{op['End Time']}\n"
            )

        improvement_suggestions = "\n## Optimization Suggestions ##\n"
        if high_wait_jobs:
            improvement_suggestions += (
                f"1. Reduce waiting for Job {high_wait_jobs[0][0]} first.\n"
            )
        low_util_machines = sorted(machine_stats.items(), key=lambda x: x[1]["utilization"])
        if low_util_machines:
            machine_id, stats = low_util_machines[0]
            improvement_suggestions += (
                f"2. Pull earlier work that activates M{machine_id} "
                f"(utilization {stats['utilization']:.1f}%).\n"
            )
        improvement_suggestions += "3. Shorten bottleneck operations in the critical path.\n"

        analysis_text = (
            f"\n{machine_analysis}{wait_analysis}{bottleneck_analysis}{improvement_suggestions}\n"
        )

    required_line = (
        f"- Total operation lines must be exactly {expected_ops}."
        if expected_ops is not None
        else "- Total operation lines must equal (num_jobs × num_machines)."
    )

    improvement_prompt = f"""
You are improving a Job Shop Scheduling solution.

Original problem:
{jssp_problem}

Current solution:
Solution:
{solution_lines}
Makespan: {calculated_makespan}

Hard constraints (must satisfy):
- Keep original machine assignments for each operation.
- No duplicate operations.
- No missing operations.
{required_line}
- Preserve per-job precedence (Op 0 -> Op 1 -> ...).

Optimization goal:
- Minimize makespan while keeping all constraints.
{analysis_text}
Output rules:
- Output only in 'Job X Operation Y, MZ' lines.
- End with exactly one line: 'Makespan: <value>'.
- Do not output explanations.
"""
    return improvement_prompt

def apply_chat_template_reflection(improvement_prompt, tokenizer):
    """
    Applies chat template specialized for improvement (reflection) stage.
    Uses a simplified system message compared to general inference.
    """
    # System message optimized for improvement stage
    reflection_system_message = """You are an expert in Job Shop Scheduling Problem (JSSP) optimization.
Based on the given initial solution and analysis information, please suggest a better schedule.

Output Format Guidelines:
1. List work sequences only in 'Job X Operation Y, MZ' format sequentially.
2. Display only 'Makespan: [value]' at the end and terminate immediately.
3. Do not repeat explanations or analysis.
4. Do not include 'Mitigation strategies' or other instructions in the output.
5. Do not write additional explanations after presenting the solution."""

    # User message uses improvement_prompt as is
    messages = [
        {"role": "system", "content": reflection_system_message},
        {"role": "user", "content": improvement_prompt}
    ]

    # Apply chat template
    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    
    return formatted_text

def generate_multiple_solutions(
    model,
    tokenizer,
    jssp_problem,
    inst_for_ortools,
    real_makespan,
    dev_map="cuda:0",
    sample=True,
    num_solutions=10,
    top_k=50,
    top_p=0.95,
    temperature=1.0,
    max_len=2048,
    reflexion_iterations=0,
    enable_improvement=False,
    schedule_save_dir=None,
    use_masking=True,
):
    """JSSP 문제에 대한 여러 솔루션을 생성하고 평가합니다."""
    print(f"솔루션 생성 시작: {num_solutions}개 요청")
    
    # 초기화
    gap_list = []
    calculated_makespan_list = []
    time_list = []
    is_feasible_list = []
    initial_solutions = []
    recalculated_solutions = []
    
    # 개선 솔루션 관련 변수
    improved_solutions = []
    improved_recalculated_solutions = []
    improved_makespan_list = []
    improved_calculated_makespan_list = []
    improved_is_feasible_list = []
    improved_gap_list = []
    
    # 0인 real_makespan 처리
    if real_makespan == 0:
        print(f"경고: real_makespan이 0입니다. 1로 대체합니다.")
        real_makespan = 1
    
    # 간소화된 메트릭스 (format_errors 제거)
    improvement_metrics = {
        "initial": {
            "min_makespan": None,
            "feasible_count": 0,
            "solutions": []
        }
    }
    
    # 프롬프트 준비
    current_prompt = jssp_problem
    prompt = apply_chat_template_inference(current_prompt, tokenizer)
    
    def _build_hooks():
        if not use_masking:
            return None, None
        fsm = build_prefix_allowed_tokens_fn_from_instance(tokenizer, inst_for_ortools)
        logits_procs = build_logits_processors(tokenizer, fsm)
        return fsm, logits_procs

    # 솔루션 생성
    for solution_index in range(num_solutions):
        print(f"솔루션 {solution_index+1}/{num_solutions} 생성 시작")
        solution_start_time = time.time()
        
        try:
            # CUDA 메모리 정리
            torch.cuda.empty_cache()
            
            # 모델을 사용한 솔루션 생성
            prefix_fn, logits_procs = _build_hooks()

            solution_outputs = gen(
                model=model, 
                prompt=prompt, 
                tokenizer=tokenizer, 
                dev_map=dev_map,
                maxlen=max_len,
                sample=sample,
                num_return_sequences=1,
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p,
                prefix_allowed_tokens_fn=prefix_fn,
                logits_processor=logits_procs,
            )
            
            solution_text = solution_outputs[0] if solution_outputs else ""
            initial_solutions.append(solution_text)
            
            # 🔍 디버깅: 첫 번째 솔루션의 LLM 원본 출력 전체 확인
            '''
            첫 번째 솔루션의 LLM 원본 출력 전체 확인하고싶으면 주석해제하고 실행
            '''
            # if solution_index == 0:
            #     print("\n" + "="*80)
            #     print("🔍 [DEBUG] LLM 원본 출력 전체:")
            #     print("="*80)
            #     print(solution_text)
            #     print("="*80)
            #     print(f"🔍 [DEBUG] 원본 출력 길이: {len(solution_text)} 문자")
            #     print("="*80 + "\n")
            
            # 작업 순서 추출 (새로운 방식만 사용)
            job_sequence = parse_solution_order(solution_text, is_debug_solution=solution_index == 0)
            
            if not job_sequence:
                print(f"솔루션 {solution_index+1}: 작업 순서 추출 실패")
                is_feasible = False
                calculated_makespan = None
                
                # 재계산된 솔루션 없음
                recalculated_solution_text = "Failed to extract operations"
                recalculated_solutions.append(recalculated_solution_text)
            else:
                # 스케줄링 정보 저장 파일 경로 설정
                schedule_file = None
                if schedule_save_dir:
                    import os
                    os.makedirs(schedule_save_dir, exist_ok=True)
                    schedule_file = f"{schedule_save_dir}/solution_{solution_index+1}_schedule.txt"
                
                # 스케줄 계산 - 개선된 알고리즘 사용
                schedule, calculated_makespan = calculate_schedule(
                    job_sequence, 
                    inst_for_ortools, 
                    is_debug_solution=solution_index == 0,
                    save_schedule_file=schedule_file
                )
                
                # 불완전한 솔루션(빈 스케줄) 체크
                if not schedule:
                    print(f"솔루션 {solution_index+1}: 불완전한 솔루션으로 skip")
                    is_feasible = False
                    calculated_makespan = None
                    
                    # 재계산된 솔루션 없음
                    recalculated_solution_text = "Incomplete solution - skipped scheduling"
                    recalculated_solutions.append(recalculated_solution_text)
                else:
                    # 🔍 디버깅: 첫 번째 솔루션의 스케줄링 결과 확인
                    if solution_index == 0:
                        print("\n" + "="*60)
                        print("🔍 [DEBUG] 계산된 스케줄 결과:")
                        print("="*60)
                        print(f"총 스케줄된 작업 수: {len(schedule)}")
                        print(f"계산된 makespan: {calculated_makespan}")
                        print("\n처음 10개 작업:")
                        for i, op in enumerate(schedule[:10]):
                            print(f"{i+1:2d}. Job {op['Job']}, Op {op['Operation']}, Machine {op['Machine']}, 시간 {op['Start Time']}-{op['End Time']}")
                        if len(schedule) > 10:
                            print(f"... (중간 {len(schedule)-20}개 생략)")
                            print("\n마지막 10개 작업:")
                            for i, op in enumerate(schedule[-10:], len(schedule)-9):
                                print(f"{i:2d}. Job {op['Job']}, Op {op['Operation']}, Machine {op['Machine']}, 시간 {op['Start Time']}-{op['End Time']}")
                        print("="*60 + "\n")
                    
                    # 유효성 검증
                    is_feasible, message = validate_schedule(schedule, inst_for_ortools, is_debug_solution=solution_index == 0)
                    
                    # 🔍 디버깅: 첫 번째 솔루션의 검증 결과 확인
                    if solution_index == 0:
                        print("\n" + "="*60)
                        print("🔍 [DEBUG] 스케줄 검증 결과:")
                        print("="*60)
                        print(f"유효성: {'✅ 유효함' if is_feasible else '❌ 무효함'}")
                        print(f"검증 메시지: {message}")
                        print("="*60 + "\n")
                    
                    # 재계산된 솔루션 포맷팅
                    recalculated_solution_text = format_solution(schedule, calculated_makespan)
                    recalculated_solutions.append(recalculated_solution_text)
                
                print(f"솔루션 {solution_index+1}: 작업 수={len(job_sequence)}, 재계산 makespan={calculated_makespan if calculated_makespan is not None else 'None (불완전)'}")
                
                # 개선 단계 실행 (enable_improvement가 True인 경우에만)
                if enable_improvement and is_feasible and calculated_makespan is not None:
                    print(f"솔루션 {solution_index+1} 개선 단계 시작")
                    
                    # 개선 프롬프트 생성
                    improvement_prompt = create_improvement_prompt(
                        recalculated_solution_text, 
                        calculated_makespan,
                        jssp_problem,
                        schedule,
                        inst_for_ortools
                    )
                    
                    # 개선 단계용 특화 템플릿 적용
                    formatted_improvement_prompt = apply_chat_template_reflection(improvement_prompt, tokenizer)
                    
                    # 개선된 솔루션 생성
                    improved_prefix_fn, improved_logits_procs = _build_hooks()

                    improved_outputs = gen(
                        model=model, 
                        prompt=formatted_improvement_prompt, 
                        tokenizer=tokenizer, 
                        dev_map=dev_map,
                        maxlen=max_len,
                        sample=sample,
                        num_return_sequences=1,
                        temperature=temperature, 
                        top_k=top_k, 
                        top_p=top_p,
                        prefix_allowed_tokens_fn=improved_prefix_fn,
                        logits_processor=improved_logits_procs,
                    )
                    
                    improved_solution_text = improved_outputs[0] if improved_outputs else ""
                    improved_solutions.append(improved_solution_text)
                    
                    # 개선된 작업 순서 추출
                    improved_job_sequence = parse_solution_order(improved_solution_text, is_debug_solution=False) # 개선 단계는 디버깅 출력 제거
                    
                    if not improved_job_sequence:
                        print(f"개선 솔루션 {solution_index+1}: 작업 순서 추출 실패")
                        improved_is_feasible = False
                        improved_calculated_makespan = None
                        
                        # 재계산된 솔루션 없음
                        improved_recalculated_solution_text = "Failed to extract improved operations"
                        improved_recalculated_solutions.append(improved_recalculated_solution_text)
                    else:
                        # 개선된 스케줄 계산
                        improved_schedule, improved_calculated_makespan = calculate_schedule(improved_job_sequence, inst_for_ortools, is_debug_solution=False)
                        
                        # 유효성 검증
                        improved_is_feasible, improved_message = validate_schedule(improved_schedule, inst_for_ortools, is_debug_solution=False)
                        
                        # 재계산된 개선 솔루션 포맷팅
                        improved_recalculated_solution_text = format_solution(improved_schedule, improved_calculated_makespan)
                        improved_recalculated_solutions.append(improved_recalculated_solution_text)
                        
                        print(f"개선 솔루션 {solution_index+1}: 작업 수={len(improved_job_sequence)}, 재계산 makespan={improved_calculated_makespan}")
                        
                        # 개선 정보 저장
                        improved_is_feasible_list.append(improved_is_feasible)
                        improved_makespan_list.append(None)  # LLM makespan 제거
                        improved_calculated_makespan_list.append(improved_calculated_makespan)
                        
                        # Gap 계산 (0으로 나누기 방지)
                        if improved_calculated_makespan is not None and real_makespan > 0:
                            improved_gap = (improved_calculated_makespan - real_makespan) / real_makespan
                            improved_gap_formatted = f"{improved_gap:.4f}"
                        else:
                            improved_gap_formatted = None
                        
                        improved_gap_list.append(improved_gap_formatted)
                        
                        # 개선 여부 확인
                        if improved_calculated_makespan is not None and calculated_makespan is not None:
                            improvement_percent = ((calculated_makespan - improved_calculated_makespan) / calculated_makespan) * 100
                            print(f"개선율: {improvement_percent:.2f}% (기존: {calculated_makespan}, 개선: {improved_calculated_makespan})")
                else:
                    print(f"솔루션 {solution_index+1} 개선 단계 생략 (enable_improvement={enable_improvement}, is_feasible={is_feasible})")
            
            elapsed_time = time.time() - solution_start_time
            
            # 결과 저장 (LLM makespan 제거)
            time_list.append(elapsed_time)
            is_feasible_list.append(is_feasible)
            
            # Gap 계산 (0으로 나누기 방지)
            if calculated_makespan is not None and real_makespan > 0:
                gap = (calculated_makespan - real_makespan) / real_makespan
                gap_formatted = f"{gap:.4f}"
            else:
                gap_formatted = None
            
            gap_list.append(gap_formatted)
            calculated_makespan_list.append(calculated_makespan)
            
            # 최소 makespan 업데이트
            if calculated_makespan is not None:
                if improvement_metrics["initial"]["min_makespan"] is None or calculated_makespan < improvement_metrics["initial"]["min_makespan"]:
                    improvement_metrics["initial"]["min_makespan"] = calculated_makespan
            
            if is_feasible:
                improvement_metrics["initial"]["feasible_count"] += 1
            
            # CUDA 메모리 정리
            del solution_outputs
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[ERROR] 솔루션 {solution_index+1} 처리 중 오류 발생: {e}")
            print(traceback.format_exc())
            
            # 오류 시 빈 결과 추가
            time_list.append(0)
            is_feasible_list.append(False)
            gap_list.append(None)
            calculated_makespan_list.append(None)
            recalculated_solutions.append("Error occurred during processing")
            
            if enable_improvement:
                improved_solutions.append("Error occurred during improvement")
                improved_recalculated_solutions.append("Error occurred during improvement")
                improved_is_feasible_list.append(False)
                improved_makespan_list.append(None)
                improved_calculated_makespan_list.append(None)
                improved_gap_list.append(None)
            
            torch.cuda.empty_cache()
    
    # 최종 결과 계산
    valid_solutions = [sol for sol in is_feasible_list if sol]
    print(f"총 유효 솔루션: {len(valid_solutions)}개")
    
    min_gap_formatted = None
    if calculated_makespan_list:
        min_makespan = min([ms for ms in calculated_makespan_list if ms is not None], default=None)
        if min_makespan is not None and real_makespan > 0:
            min_gap = (min_makespan - real_makespan) / real_makespan
            min_gap_formatted = f"{min_gap:.4f}"
            print(f"최적 솔루션 Gap: {min_gap_formatted}, 메이크스팬: {min_makespan}")
    else:
        print(f"유효한 솔루션 없음")
    
    # 개선 솔루션 결과 처리
    if enable_improvement:
        improved_valid_solutions = [sol for sol in improved_is_feasible_list if sol]
        print(f"총 유효 개선 솔루션: {len(improved_valid_solutions)}개")
        
        if improved_calculated_makespan_list:
            improved_min_makespan = min([ms for ms in improved_calculated_makespan_list if ms is not None], default=None)
            if improved_min_makespan is not None and real_makespan > 0:
                improved_min_gap = (improved_min_makespan - real_makespan) / real_makespan
                improved_min_gap_formatted = f"{improved_min_gap:.4f}"
                print(f"최적 개선 솔루션 Gap: {improved_min_gap_formatted}, 메이크스팬: {improved_min_makespan}")
                
                # 원본과 개선 솔루션 비교
                if min_makespan is not None:
                    overall_improvement = ((min_makespan - improved_min_makespan) / min_makespan) * 100
                    print(f"전체 개선율: {overall_improvement:.2f}% (기존 최적: {min_makespan}, 개선 최적: {improved_min_makespan})")
    
    print(f"솔루션 생성 완료")
    
    # 개선 솔루션 정보도 반환
    improvement_data = None
    if enable_improvement:
        improvement_data = {
            "improved_solutions": improved_solutions,
            "improved_recalculated_solutions": improved_recalculated_solutions,
            "improved_makespan_list": improved_makespan_list,
            "improved_calculated_makespan_list": improved_calculated_makespan_list,
            "improved_is_feasible_list": improved_is_feasible_list,
            "improved_gap_list": improved_gap_list
        }
    
    # LLM makespan 리스트 제거, calculated makespan만 반환
    return min_gap_formatted, is_feasible_list, gap_list, [None] * len(calculated_makespan_list), calculated_makespan_list, time_list, initial_solutions, recalculated_solutions, improvement_metrics, improvement_data 
