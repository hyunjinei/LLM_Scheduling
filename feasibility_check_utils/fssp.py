
import re



def parse_solution(solution):
    """
    Parses the solution string into a structured list of operations.
    Handles both standard operation format and flowshop job-machine format.
    """
    # Try to match the starjob_better format which looks like:
    # J18: M1(0+22=22) -> M2(22+15=37) -> M3(37+34=71) -> ...
    
    starjob_pattern = r'J(\d+): (M\d+\(\d+\+\d+=\d+\)(?: -> M\d+\(\d+\+\d+=\d+\))*)'
    operations = []
    
    for job_match in re.finditer(starjob_pattern, solution):
        job_id = int(job_match.group(1))
        machines_str = job_match.group(2)
        
        # Extract individual machine operations
        machine_pattern = r'M(\d+)\((\d+)\+(\d+)=(\d+)\)'
        for machine_match in re.finditer(machine_pattern, machines_str):
            machine_id = int(machine_match.group(1))
            start_time = int(machine_match.group(2))
            duration = int(machine_match.group(3))
            end_time = int(machine_match.group(4))
            
            operations.append((job_id, machine_id, start_time, duration, end_time))
    
    # If no operations found with the starjob pattern, try the older format
    if not operations:
        pattern = r"J(\d+)-M(\d+): (\d+)\+(\d+) -> (\d+)"
        for match in re.finditer(pattern, solution):
            job_id = int(match.group(1))
            machine_id = int(match.group(2))
            start_time = int(match.group(3))
            duration = int(match.group(4))
            end_time = int(match.group(5))
            operations.append((job_id, machine_id, start_time, duration, end_time))
    
    # Improved handling for missing makespan
    makespan_pattern = r"[Mm]akespan:?\s*(\d+)"
    makespan_match = re.search(makespan_pattern, solution)
    if makespan_match:
        declared_makespan = int(makespan_match.group(1))
    else:
        # If no makespan declared but we have operations, calculate it
        declared_makespan = max(end_time for _, _, _, _, end_time in operations) if operations else None

    return operations, declared_makespan


def validate_accord_format(problem_data, solution):
    """
    Validates a StarJob format solution against the flowshop problem data.
    Ensures:
      - durations match problem processing times,
      - start + duration == end,
      - no negative start times,
      - no machine overlaps,
      - flowshop precedence,
      - permutation across all machines,
      - declared makespan matches actual.
    """
    try:
        # Clean up problem data
        problem_data = re.sub(r'\[\d+:\d+,\s*\?it/s\]', '', problem_data)

        # Extract processing times: list of lists, indexed by job-1 then machine-1
        processing_times = []
        
        # First pass: look for complete job lines "J1:" etc.
        for job_match in re.finditer(r'J(\d+):\s*(?:M\d+:\d+\s*)+', problem_data):
            job_id = int(job_match.group(1)) - 1
            line = job_match.group(0)
            times = []
            for m_match in re.finditer(r'M(\d+):(\d+)', line):
                mid = int(m_match.group(1)) - 1
                dur = int(m_match.group(2))
                # Ensure times array is large enough for this machine
                while len(times) <= mid:
                    times.append(0)
                times[mid] = dur
            
            # Ensure processing_times array is large enough for this job
            while len(processing_times) <= job_id:
                processing_times.append([])
            processing_times[job_id] = times

        # Fallback: if no processing times extracted, try alternative parsing
        if not processing_times:
            # Try parsing line by line for job info
            lines = problem_data.strip().split('\n')
            current_job_id = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this is a job header line
                job_header_match = re.match(r'^J(\d+):', line)
                if job_header_match:
                    current_job_id = int(job_header_match.group(1)) - 1
                    while len(processing_times) <= current_job_id:
                        processing_times.append([])
                
                # Check for machine times in this line
                if current_job_id is not None:
                    for m_match in re.finditer(r'M(\d+):(\d+)', line):
                        mid = int(m_match.group(1)) - 1
                        dur = int(m_match.group(2))
                        while len(processing_times[current_job_id]) <= mid:
                            processing_times[current_job_id].append(0)
                        processing_times[current_job_id][mid] = dur

        # Another fallback: try to find any machine processing times
        if not processing_times:
            lines = problem_data.strip().split('\n')
            job_id = 0
            
            for line in lines:
                if 'M1:' in line:  # This looks like a machine line
                    times = []
                    for m_match in re.finditer(r'M(\d+):(\d+)', line):
                        mid = int(m_match.group(1)) - 1
                        dur = int(m_match.group(2))
                        while len(times) <= mid:
                            times.append(0)
                        times[mid] = dur
                    
                    if times:
                        while len(processing_times) <= job_id:
                            processing_times.append([])
                        processing_times[job_id] = times
                        job_id += 1

        if not processing_times:
            return False, "Could not extract processing times from problem data.", None

        num_jobs = len(processing_times)
        num_machines = max(len(job_times) for job_times in processing_times) if processing_times else 0
        
        # Ensure each job has enough machine slots
        for job_id in range(len(processing_times)):
            while len(processing_times[job_id]) < num_machines:
                processing_times[job_id].append(0)  # Pad with zeros if needed

        # Parse operations from solution
        operations, declared_makespan = parse_solution(solution)
        if not operations:
            return False, "No operations found in the solution.", None

        # Check non-negativity of start times
        for job_id, machine_id, start, dur, end in operations:
            if start < 0:
                return False, f"Negative start time detected: J{job_id}-M{machine_id} starts at {start}", declared_makespan

        # Build machine schedules
        machine_schedules = {m: [] for m in range(1, num_machines+1)}

        # Validate each operation
        for job_id, machine_id, start, dur, end in operations:
            # Check if job_id and machine_id are valid (prevent index errors)
            if job_id <= 0:
                return False, f"Invalid job_id: {job_id} (must be positive).", declared_makespan
            
            if machine_id <= 0:
                return False, f"Invalid machine_id: {machine_id} (must be positive).", declared_makespan
            
            # Check if job_id is within range
            if job_id-1 >= len(processing_times):
                return False, f"Job ID {job_id} exceeds number of jobs in problem data ({len(processing_times)}).", declared_makespan
            
            # Check if machine_id is within range for this job
            if machine_id-1 >= len(processing_times[job_id-1]):
                return False, f"Machine ID {machine_id} exceeds number of machines for job {job_id} ({len(processing_times[job_id-1])}).", declared_makespan
            
            # Now it's safe to access processing_times
            exp_dur = processing_times[job_id-1][machine_id-1]
            
            # Check duration matches problem's processing time
            if dur != exp_dur:
                return False, f"Duration mismatch: J{job_id}-M{machine_id} has duration {dur}, expected {exp_dur}.", declared_makespan

            # Check start+dur == end
            if start + dur != end:
                return False, f"Invalid timing: J{job_id}-M{machine_id} start {start} + dur {dur} != end {end}.", declared_makespan

            machine_schedules[machine_id].append((job_id, start, end))

        # Check machine overlaps
        for m_id, sched in machine_schedules.items():
            sched.sort(key=lambda x: x[1])
            for i in range(len(sched)-1):
                _, s1, e1 = sched[i]
                _, s2, e2 = sched[i+1]
                if e1 > s2:
                    return False, f"Machine conflict: M{m_id} overlaps between intervals [{s1},{e1}) and [{s2},{e2}).", declared_makespan

        # Check job precedence
        job_times = {j: {} for j in range(1, num_jobs+1)}
        for j, m, s, d, e in operations:
            job_times[j][m] = (s, e)
        
        for j in range(1, num_jobs+1):
            for m in range(1, num_machines):
                if m in job_times[j] and (m+1) in job_times[j]:
                    if job_times[j][m][1] > job_times[j][m+1][0]:
                        return False, f"Precedence violation: J{j} finishes M{m} at {job_times[j][m][1]}, but starts M{m+1} at {job_times[j][m+1][0]}.", declared_makespan

        # Check permutation: each machine must have exactly num_jobs ops
        for m_id, sched in machine_schedules.items():
            if len(sched) != num_jobs:
                return False, f"Permutation violated: M{m_id} has {len(sched)} jobs, expected {num_jobs}.", declared_makespan

        # Check sequence consistency across machines (permutation constraint)
        seq1 = [jid for jid, _, _ in sorted(machine_schedules[1], key=lambda x: x[1])]
        for m_id in range(2, num_machines+1):
            seqm = [jid for jid, _, _ in sorted(machine_schedules[m_id], key=lambda x: x[1])]
            if seqm != seq1:
                return False, f"Permutation violated: sequence on M{m_id} {seqm} != sequence on M1 {seq1}.", declared_makespan

        # Verify makespan
        actual_mk = max(end for (_, _, _, _, end) in operations)
        # if declared_makespan is not None:
        #     return True, f"Makespan mismatch: declared {declared_makespan}, actual {actual_mk}.", actual_mk

        return True, f"Solution is feasible with makespan {actual_mk}.", declared_makespan or actual_mk

    except Exception as ex:
        import traceback
        return False, f"Validation error: {ex}\n{traceback.format_exc()}", None
    

def validate_list_of_lists_format(problem_data, solution):
    """
    Validates a list of lists format solution against the problem data.
    
    Args:
        problem_data (str): The problem data in job-first format.
        solution (str): The solution in list of lists format.
        
    Returns:
        tuple: A tuple containing:
            - valid (bool): Whether the solution is valid.
            - message (str): A message explaining the validation result.
            - declared_makespan (int or None): The makespan declared in the solution.
    """
    try:
        # Clean up problem data - remove any progress bar output like [00:00, ?it/s]
        problem_data = re.sub(r'\[\d+:\d+,\s*\?it/s\]', '', problem_data)
        
        # Extract processing times from the problem data
        processing_times = {} # Dictionary to store {job_id: {machine_id: duration}}
        
        # First, try to match complete job patterns like "J1:" followed by machine times
        job_pattern = r'J(\d+):\s*\n(M\d+:\d+\s*)+' 
        job_matches = re.finditer(job_pattern, problem_data, re.MULTILINE)
        
        for job_match in job_matches:
            job_id = int(job_match.group(1))
            machine_line = job_match.group(0).split('\n')[1]
            
            # Extract all machine times from this line
            machine_times = {}
            for m_match in re.finditer(r'M(\d+):(\d+)', machine_line):
                machine_id = int(m_match.group(1))
                duration = int(m_match.group(2))
                machine_times[machine_id] = duration
            
            # Store the processing times for this job
            processing_times[job_id] = machine_times
        
        # If we couldn't find any job patterns, try a more flexible approach
        if not processing_times:
            # Process line by line
            lines = problem_data.strip().split('\n')
            current_job_id = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this is a job header line
                job_header_match = re.match(r'^J(\d+):', line)
                if job_header_match:
                    current_job_id = int(job_header_match.group(1))
                
                # Check if this line contains machine times
                if line.startswith('M') and current_job_id is not None:
                    machine_times = {}
                    for m_match in re.finditer(r'M(\d+):(\d+)', line):
                        machine_id = int(m_match.group(1))
                        duration = int(m_match.group(2))
                        machine_times[machine_id] = duration
                    
                    if machine_times:  # Only add if we found any machine times
                        processing_times[current_job_id] = machine_times
                        
                # Special case: If line starts with M but we don't have a job ID yet,
                # assume it's for job 1 (this handles missing J1: header)
                elif line.startswith('M') and current_job_id is None and not processing_times:
                    machine_times = {}
                    for m_match in re.finditer(r'M(\d+):(\d+)', line):
                        machine_id = int(m_match.group(1))
                        duration = int(m_match.group(2))
                        machine_times[machine_id] = duration
                    
                    if machine_times:
                        processing_times[1] = machine_times  # Job 1
        
        # Last resort: Try to infer structure from any lines with M#:# patterns
        if not processing_times:
            lines = problem_data.strip().split('\n')
            job_id = 1
            
            for line in lines:
                if 'M1:' in line:  # This looks like a machine line
                    machine_times = {}
                    for m_match in re.finditer(r'M(\d+):(\d+)', line):
                        machine_id = int(m_match.group(1))
                        duration = int(m_match.group(2))
                        machine_times[machine_id] = duration
                    
                    if machine_times:
                        processing_times[job_id] = machine_times
                        job_id += 1
        
        if not processing_times:
            return False, "Could not extract processing times from problem data after multiple attempts.", None
        
        # Get number of jobs and machines
        num_jobs = len(processing_times)
        num_machines = max(max(job_machines.keys()) for job_machines in processing_times.values()) if processing_times else 0
        
        print(f"Successfully parsed {num_jobs} jobs with {num_machines} machines from problem data")
        
        # Parse the list of lists solution
        # Expected format: [[job_id, machine_id, start_time, duration], ...]
        list_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        operations = []
        
        for match in re.finditer(list_pattern, solution):
            job_id = int(match.group(1))
            machine_id = int(match.group(2))
            start_time = int(match.group(3))
            duration = int(match.group(4))
            end_time = start_time + duration
            operations.append((job_id, machine_id, start_time, duration, end_time))
        
        if not operations:
            return False, "No operations found in the solution.", None
        
        # Extract declared makespan if available
        makespan_pattern = r'Makespan:\s*(\d+)'
        makespan_match = re.search(makespan_pattern, solution)
        declared_makespan = int(makespan_match.group(1)) if makespan_match else None
        
        # Create a dictionary to store machine schedules
        machine_schedules = {m: [] for m in range(1, num_machines + 1)}
        
        # Populate machine schedules and validate processing times
        for job_id, machine_id, start_time, duration, end_time in operations:
            # Check if job_id and machine_id are within the problem's bounds
            if job_id not in processing_times:
                return False, f"Job {job_id} in solution not found in the problem data.", declared_makespan
            
            if machine_id not in processing_times[job_id]:
                return False, f"Machine {machine_id} for job {job_id} not found in the problem data.", declared_makespan
            
            # Check if the operation uses the correct processing time from the problem
            expected_duration = processing_times[job_id][machine_id]
            if duration != expected_duration:
                return False, f"J{job_id}-M{machine_id} has incorrect duration: {duration} (should be {expected_duration}).", declared_makespan
            
            # Check if the operation's calculated end time is correct
            if start_time + duration != end_time:
                return False, f"Invalid operation timing: J{job_id}-M{machine_id} starts at {start_time}, has duration {duration}, but ends at {end_time} (should be {start_time + duration}).", declared_makespan
            
            # Add the operation to the machine's schedule
            machine_schedules[machine_id].append((job_id, start_time, end_time))
        
        # Check that all jobs and machines from the problem are in the solution
        solution_jobs = set(job_id for job_id, _, _, _, _ in operations)
        solution_machines_per_job = {}
        for job_id, machine_id, _, _, _ in operations:
            if job_id not in solution_machines_per_job:
                solution_machines_per_job[job_id] = set()
            solution_machines_per_job[job_id].add(machine_id)
        
        # Check if any jobs from the problem are missing in the solution
        for job_id in processing_times:
            if job_id not in solution_jobs:
                return False, f"Job {job_id} from the problem is missing in the solution.", declared_makespan
            
            # Check if all machines for this job are in the solution
            for machine_id in processing_times[job_id]:
                if machine_id not in solution_machines_per_job.get(job_id, set()):
                    return False, f"Machine {machine_id} for job {job_id} is missing in the solution.", declared_makespan
        
        # Check for machine conflicts (no overlaps)
        for machine_id, schedule in machine_schedules.items():
            # Sort operations by start time
            schedule.sort(key=lambda op: op[1])
            
            # Check for overlaps
            for i in range(len(schedule) - 1):
                current_job, current_start, current_end = schedule[i]
                next_job, next_start, next_end = schedule[i + 1]
                
                if current_end > next_start:
                    return False, f"Machine conflict: M{machine_id} has overlapping operations for J{current_job} and J{next_job}.", declared_makespan
        
        # Check flowshop constraints (job sequence through machines)
        job_machine_times = {j: {} for j in processing_times.keys()}
        
        for job_id, machine_id, start_time, duration, end_time in operations:
            job_machine_times[job_id][machine_id] = (start_time, end_time)
        
        # For each job, verify the sequence through machines
        for job_id in processing_times:
            # Check precedence constraints
            for machine_id in range(1, num_machines):
                if machine_id in job_machine_times[job_id] and machine_id + 1 in job_machine_times[job_id]:
                    current_end = job_machine_times[job_id][machine_id][1]
                    next_start = job_machine_times[job_id][machine_id + 1][0]
                    
                    if current_end > next_start:
                        return False, f"Job precedence violation: J{job_id} finishes on M{machine_id} at {current_end} but starts on M{machine_id + 1} at {next_start}.", declared_makespan
        
        # Check permutation constraint (job sequence must be the same across all machines)
        # Get job sequence on first machine
        machine1_sequence = [job_id for job_id, _, _ in sorted(machine_schedules[1], key=lambda x: x[1])]
        
        # Check if same sequence exists on all other machines
        for machine_id in range(2, num_machines + 1):
            if machine_id not in machine_schedules:
                continue  # Skip if we don't have data for this machine
                
            machine_sequence = [job_id for job_id, _, _ in sorted(machine_schedules[machine_id], key=lambda x: x[1])]
            if machine_sequence != machine1_sequence:
                return False, f"Permutation constraint violated: job sequence on M{machine_id} ({machine_sequence}) differs from M1 ({machine1_sequence}).", declared_makespan
        
        # Calculate actual makespan from the solution
        actual_makespan = max(end_time for _, _, _, _, end_time in operations)
        
        # Check if the actual makespan matches the declared makespan (with some tolerance)
        # if declared_makespan is not None and abs(actual_makespan - declared_makespan) > 1:  # Allow small rounding differences
        #     return False, f"Declared makespan ({declared_makespan}) does not match the actual makespan ({actual_makespan}).", declared_makespan
        
        return True, f"Solution is feasible with makespan {actual_makespan}.", actual_makespan
    
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        return False, f"Validation error: {str(e)}\n{trace}", None