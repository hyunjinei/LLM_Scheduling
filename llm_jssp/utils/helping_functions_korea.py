# helping_functions.py

import csv 
import os
import json
import traceback
import sys
import time


def print_number_of_trainable_model_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    try:
        trainable_model_params = 0
        all_model_params = 0
        
        # 각 매개변수 그룹별 정보 수집
        param_groups = {}
        
        for name, param in model.named_parameters():
            all_model_params += param.numel()
            
            # 매개변수 그룹 수집
            group = name.split('.')[0] if '.' in name else 'other'
            if group not in param_groups:
                param_groups[group] = {'total': 0, 'trainable': 0}
            
            param_groups[group]['total'] += param.numel()
            
            if param.requires_grad:
                trainable_model_params += param.numel()
                param_groups[group]['trainable'] += param.numel()
        
        # 그룹별 세부정보 출력 - 디버그 메시지 제거
        # 전체 요약 출력
        trainable_percent = (trainable_model_params / all_model_params * 100) if all_model_params > 0 else 0
        print(f"총 학습가능 매개변수: {trainable_model_params:,}개")
        print(f"총 매개변수: {all_model_params:,}개")
        print(f"학습가능 매개변수 비율: {trainable_percent:.2f}%")
        
        return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"
    
    except Exception as e:
        print(f"매개변수 정보 계산 중 오류 발생: {str(e)}")
        return f"Error calculating parameters: {str(e)}"

def write_csv(results, filename='llm_jssp_results.csv'):
    """
    Writes the results to a CSV file.
    """
    try:
        # 디렉토리 생성
        dir_path = os.path.dirname(filename)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # CSV 파일 작성
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['num_jobs','num_machines', 'best_gap', 'is_feasible_list', 'gap_list', 'time_list', 'llm_makespan_list', 'calculated_makespan_list', 'peft_model_text_output'])
            
            for result in results:
                writer.writerow(result)
        
        print(f"Results successfully saved to {filename}")
    
    except Exception as e:
        print(f"CSV 파일 '{filename}' 저장 실패: {str(e)}")
        print(f"Failed to write CSV file {filename}. Error: {e}")


# def save_results(results, start, num_solutions, temperature, top_p, top_k, filepath=None):
#     if not filepath:
#         filepath = f"./validation_{start}_results_num_sol_{num_solutions}_t_{temperature}_p_{top_p}_k_{top_k}.csv"
    
#     try:
#         with open(filepath, mode="w", encoding="utf-8", newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(["num_jobs", "num_machines", "best_gap", "is_feasible_list", "gap_list", "time_list", "llm_makespan_list", "calculated_makespan_list", "peft_model_text_output"])
#             for row in results:
#                 writer.writerow(row)
#         print(f"[DEBUG] Results successfully saved to {filepath}.")
#     except Exception as e:
#         print(f"[ERROR] Failed to write CSV file {filepath}. Error: {e}")

def save_results(results, start, num_solutions, temperature, top_p, top_k, timestamp=None, reflexion_iterations=0):
    """
    모델이 생성한 결과를 CSV 파일로 저장합니다.
    """
    try:
        # 결과가 비어있으면 리턴
        if not results:
            print("결과가 비어있어 저장 취소")
            return
        
        # 현재 시간을 파일명에 추가
        if timestamp is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # CSV 파일명 생성
        filename = f'./validation_{start}_results_num_sol_{num_solutions}_t_{temperature}_p_{top_p}_k_{top_k}_{timestamp}.csv'
        print(f"결과 저장 중: {filename} (항목 수: {len(results)})")
        
        write_csv(results, filename=filename)
        
        print(f"결과가 성공적으로 저장되었습니다: {filename}")
    except Exception as e:
        print(f"결과 저장 중 오류 발생: {str(e)}")
        traceback.print_exc()

def save_initial_solutions_csv(solutions, start, num_solutions, temperature, top_p, top_k):
    """
    초기 솔루션 목록을 CSV 파일로 저장합니다.
    """
    # print(f"[DEBUG:DETAIL] save_initial_solutions_csv 시작")  # 디버그 메시지 비활성화
    # print(f"[DEBUG:DETAIL] 파라미터: start={start}, num_solutions={num_solutions}, temperature={temperature}, top_p={top_p}, top_k={top_k}")  # 디버그 메시지 비활성화
    # print(f"[DEBUG:DETAIL] 솔루션 항목 수: {len(solutions)}")  # 디버그 메시지 비활성화
    
    try:
        # 현재 시간을 파일명에 추가
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        filename = f'./validation_{start}_initial_solutions_num_sol_{num_solutions}_t_{temperature}_p_{top_p}_k_{top_k}_{timestamp}.csv'
        # print(f"[DEBUG:DETAIL] 저장 파일명: '{filename}'")  # 디버그 메시지 비활성화
        
        # 데이터 변환 (솔루션 목록 펼치기)
        solutions_flat = []
        
        # 각 문제별 솔루션 처리
        for prob_idx, prob_sol in enumerate(solutions):
            # print(f"[DEBUG:DETAIL] 문제 {prob_idx+1}/{len(solutions)} 처리 중")  # 디버그 메시지 비활성화
            
            # 필수 키 확인
            required_keys = ['problem_index', 'num_jobs', 'num_machines', 'real_makespan', 'initial_solutions']
            missing_keys = [key for key in required_keys if key not in prob_sol]
            
            if missing_keys:
                # print(f"[DEBUG:WARN] 문제 {prob_idx+1}에 필수 키 없음: {missing_keys}")  # 디버그 메시지 비활성화
                continue
                
            problem_index = prob_sol['problem_index']
            initial_solutions = prob_sol.get('initial_solutions', [])
            
            # print(f"[DEBUG:DETAIL] 문제 {problem_index}: 초기 솔루션 수 {len(initial_solutions)}")  # 디버그 메시지 비활성화
            
            # 각 솔루션 처리
            for i, sol in enumerate(initial_solutions):
                solution_info = [
                    problem_index, 
                    prob_sol['num_jobs'], 
                    prob_sol['num_machines'],
                    prob_sol['real_makespan'],
                    i + 1,  # 솔루션 번호
                    sol  # 솔루션 텍스트
                ]
                solutions_flat.append(solution_info)
                
                # 첫 번째 솔루션만 샘플로 출력
                if i == 0:
                    sol_text = sol[:100] + "..." if len(sol) > 100 else sol
                    # print(f"[DEBUG:DETAIL] 문제 {problem_index}, 솔루션 1 샘플: {sol_text}")  # 디버그 메시지 비활성화
                
        # print(f"[DEBUG:DETAIL] 총 펼친 솔루션 수: {len(solutions_flat)}")  # 디버그 메시지 비활성화
                
        # CSV 파일 저장
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Problem Index', 'Num Jobs', 'Num Machines', 'Real Makespan', 'Solution Number', 'Solution Text'])
            
            for i, row in enumerate(solutions_flat):
                # print(f"[DEBUG:DETAIL] 솔루션 {i+1}/{len(solutions_flat)} 저장 중")  # 디버그 메시지 비활성화
                writer.writerow(row)
        
        # print(f"[DEBUG:DETAIL] 초기 솔루션 CSV 저장 완료: {filename}")  # 디버그 메시지 비활성화
        print(f"초기 솔루션이 성공적으로 저장되었습니다: {filename}")
    
    except Exception as e:
        print(f"[ERROR] 초기 솔루션 CSV 저장 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        print(f"초기 솔루션 CSV 저장 실패: {e}")
    
    # print(f"[DEBUG:DETAIL] save_initial_solutions_csv 완료\n")  # 디버그 메시지 비활성화
    
def save_problem_solutions(problem_id, solutions_data, real_makespan, timestamp=None, is_reflection=False):
    """
    한 문제에 대한 모든 솔루션을 별도 CSV 파일에 저장합니다.
    
    Args:
        problem_id (int): 문제 ID (파일명에 사용)
        solutions_data (dict): 솔루션 데이터 (다음 키를 포함해야 함)
            - 'original': 원본 솔루션 텍스트 리스트
            - 'recalculated': 재계산된 솔루션 텍스트 리스트
            - 'original_makespan': 원본 makespan 리스트
            - 'recalculated_makespan': 재계산된 makespan 리스트
            - 'is_feasible': 유효성 여부 리스트
            - 'gap': Gap 비율 리스트
        real_makespan (float): 실제 최적 makespan 값
        timestamp (str, optional): 타임스탬프
        is_reflection (bool): 개선 솔루션 여부
    """
    try:
        # 타임스탬프 생성
        if timestamp is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # 0인 real_makespan 처리
        if real_makespan == 0:
            print(f"경고: 문제 {problem_id}의 real_makespan이 0입니다. 1로 대체합니다.")
            real_makespan = 1
        
        # 파일명 생성 (개선 솔루션 여부에 따라 다른 파일명 사용)
        if is_reflection:
            filename = f"{problem_id}_fix_{timestamp}_reflection.csv"
            print(f"문제 {problem_id} 개선 솔루션 저장 중: {filename}")
        else:
            filename = f"{problem_id}_fix_{timestamp}.csv"
            print(f"문제 {problem_id} 솔루션 저장 중: {filename}")
        
        # 디렉토리 확인
        dir_path = os.path.dirname(filename)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # 데이터 확인
        if not solutions_data or 'original' not in solutions_data or 'recalculated' not in solutions_data:
            print(f"경고: 문제 {problem_id}의 솔루션 데이터가 비어있거나 필요한 키가 없습니다.")
            return
        
        # CSV 파일 작성
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Solution_Number', 
                'Original_Makespan', 
                'Recalculated_Makespan', 
                'Gap_vs_Optimal', 
                'Is_Valid', 
                'Original_Solution', 
                'Recalculated_Solution'
            ])
            
            # 솔루션 수 (최대 길이 기준)
            max_sols = max(
                len(solutions_data.get('original', [])),
                len(solutions_data.get('recalculated', [])),
                len(solutions_data.get('original_makespan', [])),
                len(solutions_data.get('recalculated_makespan', []))
            )
            
            # 데이터가 없는 경우 처리
            if max_sols == 0:
                print(f"경고: 문제 {problem_id}에 대한 솔루션이 없습니다.")
                return
            
            # 각 솔루션별로 행 작성
            for i in range(max_sols):
                # 안전하게 데이터 추출
                orig_sol = solutions_data.get('original', [])[i] if i < len(solutions_data.get('original', [])) else ""
                recalc_sol = solutions_data.get('recalculated', [])[i] if i < len(solutions_data.get('recalculated', [])) else ""
                orig_ms = solutions_data.get('original_makespan', [])[i] if i < len(solutions_data.get('original_makespan', [])) else None
                recalc_ms = solutions_data.get('recalculated_makespan', [])[i] if i < len(solutions_data.get('recalculated_makespan', [])) else None
                is_feasible = solutions_data.get('is_feasible', [])[i] if i < len(solutions_data.get('is_feasible', [])) else False
                
                # 0 makespan 처리
                if orig_ms == 0:
                    orig_ms = "Invalid(0)"
                
                # Gap 계산
                if recalc_ms is not None and real_makespan > 0:
                    gap = (recalc_ms - real_makespan) / real_makespan
                    gap_str = f"{gap:.4f}"
                else:
                    gap_str = "N/A"
                
                # CSV 행 작성
                writer.writerow([
                    i+1,  # 솔루션 번호
                    orig_ms,  # 원본 makespan
                    recalc_ms,  # 재계산 makespan
                    gap_str,  # gap
                    "Yes" if is_feasible else "No",  # 유효성
                    orig_sol,  # 원본 솔루션 텍스트
                    recalc_sol  # 재계산 솔루션 텍스트
                ])
        
        sol_type = "개선" if is_reflection else "초기"
        print(f"문제 {problem_id}의 {sol_type} 솔루션 {max_sols}개가 성공적으로 저장되었습니다: {filename}")
        
    except Exception as e:
        sol_type = "개선" if is_reflection else "초기"
        print(f"문제 {problem_id} {sol_type} 솔루션 저장 중 오류 발생: {str(e)}")
        traceback.print_exc()

def save_raw_model_outputs(raw_outputs, start, num_solutions, temperature, top_p, top_k, timestamp=None, is_reflection=False):
    """
    모델이 생성한 원시 출력을 텍스트 파일로 저장합니다.
    
    Args:
        raw_outputs: 원시 모델 출력 리스트
        start: 시작 인덱스
        num_solutions: 솔루션 수
        temperature: 온도 파라미터
        top_p: top_p 파라미터
        top_k: top_k 파라미터
        timestamp: 타임스탬프
        is_reflection: 개선 솔루션 여부
    """
    # print(f"\n[디버그:상세] save_raw_model_outputs 시작")  # 디버그 메시지 비활성화
    # print(f"[디버그:상세] 파라미터: start={start}, num_solutions={num_solutions}, temperature={temperature}, top_p={top_p}, top_k={top_k}")  # 디버그 메시지 비활성화
    
    try:
        # 출력이 비어있으면 리턴
        if not raw_outputs:
            # print("[디버그:상세] 원시 출력이 비어있어 저장 취소")  # 디버그 메시지 비활성화
            return
        
        # 출력 내용 분석
        # print(f"[디버그:상세] 원시 출력 항목 수: {len(raw_outputs)}")  # 디버그 메시지 비활성화
        for i, output in enumerate(raw_outputs):
            # print(f"[디버그:상세] 항목 {i+1} 키: {list(output.keys())}")  # 디버그 메시지 비활성화
            if 'raw_outputs' in output:
                outputs_list = output['raw_outputs']
                # print(f"[디버그:상세] 항목 {i+1} 출력 수: {len(outputs_list)}")  # 디버그 메시지 비활성화
                for j, text in enumerate(outputs_list):
                    # print(f"[디버그:상세] 항목 {i+1}, 출력 {j+1} 타입: {type(text).__name__}")  # 디버그 메시지 비활성화
                    # print(f"[디버그:상세] 항목 {i+1}, 출력 {j+1} 길이: {len(text)}")  # 디버그 메시지 비활성화
                    # print(f"[디버그:상세] 항목 {i+1}, 출력 {j+1} 샘플: '{text[:50]}...'")  # 디버그 메시지 비활성화
                    pass
        
        # 현재 시간을 파일명에 추가
        if timestamp is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # 텍스트 파일명 생성 (개선 솔루션 여부에 따라 다른 파일명 사용)
        if is_reflection:
            filename = f'./validation_{start}_raw_outputs_num_sol_{num_solutions}_t_{temperature}_p_{top_p}_k_{top_k}_{timestamp}_reflection.txt'
            # print(f"[디버그:상세] 개선된 원시 출력 저장 파일명: '{filename}'")  # 디버그 메시지 비활성화
        else:
            filename = f'./validation_{start}_raw_outputs_num_sol_{num_solutions}_t_{temperature}_p_{top_p}_k_{top_k}_{timestamp}.txt'
            # print(f"[디버그:상세] 원시 출력 저장 파일명: '{filename}'")  # 디버그 메시지 비활성화
        
        # 디렉토리 확인/생성
        dirname = os.path.dirname(filename)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
            # print(f"[디버그:상세] 디렉토리 {dirname} 생성 완료")  # 디버그 메시지 비활성화
        else:
            # print(f"[디버그:상세] 디렉토리 {dirname if dirname else '.'} 확인/생성 완료")  # 디버그 메시지 비활성화
            pass
        
        # 파일 저장
        start_time = time.time()
        with open(filename, 'w', encoding='utf-8') as f:
            for i, output in enumerate(raw_outputs):
                # print(f"[디버그:상세] 문제 {i+1}/{len(raw_outputs)} 처리 중")  # 디버그 메시지 비활성화
                
                # 문제 메타데이터 저장
                problem_index = output.get('problem_index', i)
                num_jobs = output.get('num_jobs', 'N/A')
                num_machines = output.get('num_machines', 'N/A')
                real_makespan = output.get('real_makespan', 'N/A')
                
                sol_type = "개선" if is_reflection else "초기"
                f.write(f"=== 문제 {problem_index} ({sol_type} 솔루션, Job 수: {num_jobs}, Machine 수: {num_machines}, 실제 Makespan: {real_makespan}) ===\n\n")
                
                # 원시 출력 저장
                if 'raw_outputs' in output:
                    outputs_list = output['raw_outputs']
                    # print(f"[디버그:상세] 문제 {problem_index}: 출력 수={len(outputs_list)}")  # 디버그 메시지 비활성화
                    
                    for j, text in enumerate(outputs_list):
                        # print(f"[디버그:상세] 문제 {problem_index}, 출력 {j+1}/{len(outputs_list)} 저장 중 (길이: {len(text)})")  # 디버그 메시지 비활성화
                        f.write(f"--- Solution {j+1} ---\n")
                        f.write(f"{text}\n\n")
                else:
                    # print(f"[디버그:상세] 문제 {problem_index}: 출력 없음")  # 디버그 메시지 비활성화
                    f.write("--- 출력 없음 ---\n\n")
        
        end_time = time.time()
        
        # 파일 정보 출력
        file_size = os.path.getsize(filename) / 1024
        sol_type = "개선된" if is_reflection else "원시"
        # print(f"[디버그:상세] 저장된 파일 크기: {file_size:.2f} KB")  # 디버그 메시지 비활성화
        
        # 파일 내용 미리보기
        with open(filename, 'r', encoding='utf-8') as f:
            preview_lines = [next(f, None) for _ in range(10)]
            preview_text = "".join([f"  {line}" for line in preview_lines if line])
            # print(f"[디버그:상세] 저장된 파일 첫 10줄:\n{preview_text}")  # 디버그 메시지 비활성화
        
        # print(f"[디버그:상세] save_raw_model_outputs 완료")  # 디버그 메시지 비활성화
        # print()  # 디버그 메시지 비활성화
        
    except Exception as e:
        sol_type = "개선된" if is_reflection else "원시"
        print(f"{sol_type} 모델 출력 저장 중 오류 발생: {str(e)}")
        traceback.print_exc()

def save_improvement_metrics(metrics, start, num_solutions, temperature, top_p, top_k, timestamp=None, reflexion_iterations=0):
    """간소화된 개선 메트릭스 저장 함수 (호환성을 위해 유지)"""
    # print(f"\n[DEBUG:DETAIL] save_improvement_metrics 시작")  # 디버그 메시지 비활성화
    # print(f"[DEBUG:DETAIL] 파라미터: start={start}, num_solutions={num_solutions}, temperature={temperature}, top_p={top_p}, top_k={top_k}")  # 디버그 메시지 비활성화
    # print(f"[DEBUG:DETAIL] 개선 로직 제거됨 - 메트릭스 저장 생략")  # 디버그 메시지 비활성화
    # print(f"[DEBUG:DETAIL] save_improvement_metrics 완료")  # 디버그 메시지 비활성화
    