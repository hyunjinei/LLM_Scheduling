
from io import StringIO
import numpy as np
import re


def read_matrix_form_jssp(matrix_content: str, sep: str = " "):
    """
    Reads the JSSP (Job Shop Scheduling Problem) problem from a string and returns problem data in various formats.

    Args:
        matrix_content (str): A string containing the JSSP problem in matrix form.
        sep (str, optional): The separator used in the matrix content. Defaults to ' '.

    Returns:
        tuple: A tuple containing:
            - n (int): Number of jobs.
            - m (int): Number of machines.
            - inst_for_ortools (list): Instance formatted for OR-Tools.
            - ms (float or None): Makespan if available, else None.
            - sol (numpy.ndarray or None): Solution if available, else None.
            - machine_to_tasks (dict): Dictionary mapping machines to tasks.
    """
    f = StringIO(matrix_content)

    # Load the shape and instance
    n, m = map(int, next(f).split(sep))
    instance = np.array(
        [line.split(sep) for line in (next(f).strip() for _ in range(n))], dtype=np.int16
    )
    inst_for_ortools = instance.reshape((n, m, 2))

    # Load makespan (if available)
    ms = None
    try:
        ms = float(next(f).strip())
    except (StopIteration, ValueError):
        pass

    # Load solution (if available)
    solution_lines = [line.strip() for line in iter(f.readline, "")]
    sol = (
        np.array([line.split(sep) for line in solution_lines], dtype=np.int32)
        if solution_lines
        else None
    )

    # Create machine_to_tasks dictionary
    initial_operation_matrix = np.arange(n * m).reshape((n, m))
    machine_to_tasks = {}
    if sol is not None:
        for machine_index, machine_row in enumerate(sol):
            tasks_list = []
            for job_value in machine_row:
                indices = np.where(initial_operation_matrix == job_value)
                if indices[0].size > 0:
                    tasks_list.append((indices[0][0], indices[1][0]))
            machine_to_tasks[machine_index] = tasks_list

    return n, m, inst_for_ortools.tolist(), ms, sol, machine_to_tasks


def parse_problem_data(problem_data):
    """
    Parses the job data into a structured format.

    Args:
        problem_data (str): The problem data as a string.

    Returns:
        dict or tuple: If successful, returns a dictionary with job IDs as keys and lists of (machine, duration) tuples as values.
                       If an error occurs, returns a tuple (False, error_message).
    """
    jobs = {}
    current_job = None
    for line in problem_data.splitlines():
        line = line.strip()  # Remove any leading/trailing spaces

        if not line:  # Skip empty lines
            continue

        if line.startswith("J"):  # Job identifier
            current_job = line[:-1]
            jobs[current_job] = []

        elif line.startswith("M") and current_job:  # Machine and duration line
            machine_assignments = line.split()  # Split based on any whitespace
            for assignment in machine_assignments:
                try:
                    # Correctly parse machine and duration (e.g., M0:48 -> machine 0, duration 48)
                    machine, duration = map(int, assignment[1:].split(":"))
                    jobs[current_job].append((machine, duration))
                except ValueError:
                    return False, f"Invalid format in line: {assignment}"

    return jobs


def parse_solution(solution):
    """
    Parses the solution string into a structured list of operations.

    Args:
        solution (str): The solution string.

    Returns:
        tuple: A tuple containing:
            - operations (list): A list of operations in the format (job_id, machine_id, start_time, duration, end_time).
            - declared_makespan (int or None): The makespan declared in the solution, or None if not found.
    """
    # Pattern to extract operations
    pattern = r"J(\d+)-M(\d+): (\d+)\+(\d+) -> (\d+)"
    operations = []

    for match in re.finditer(pattern, solution):
        job_id = int(match.group(1))
        machine_id = int(match.group(2))
        start_time = int(match.group(3))
        duration = int(match.group(4))
        end_time = int(match.group(5))
        operations.append((job_id, machine_id, start_time, duration, end_time))

    # Improved handling for missing makespan
    makespan_pattern = r"Makespan: (\d+)"
    makespan_match = re.search(makespan_pattern, solution)
    if makespan_match:
        declared_makespan = int(makespan_match.group(1))
    else:
        declared_makespan = None  # Return None for makespan if not found

    return operations, declared_makespan


def validate_accord_format(problem_data, solution):
    jobs = parse_problem_data(problem_data)

    if isinstance(jobs, tuple):  # Check if an error was encountered during parsing
        return False, f"Error parsing problem data: {jobs[1]}", None  # Return error message if any

    if not solution.strip():
        return False, "Solution string is empty.", None

    operations, declared_makespan = parse_solution(solution)

    if not operations:
        return False, "No valid operations found in the solution.", declared_makespan

    # Ensure all jobs are accounted for
    solution_jobs = set(f"J{job_id}" for job_id, _, _, _, _ in operations)
    if solution_jobs != set(jobs.keys()):
        return (
            False,
            f"Missing jobs in the solution. Expected: {set(jobs.keys())}, Found: {solution_jobs}",
            declared_makespan,
        )

    # Create dictionaries to track the last end time for each machine and job
    machine_end_times = {}
    job_end_times = {}
    job_next_operation_indices = {job_id: 0 for job_id in jobs}

    # Prepare to track operations for each machine to detect overlaps
    machine_operations = {}

    try:
        # Sort operations by their start time to process them in chronological order
        operations.sort(key=lambda x: x[2])

        for job_id, machine_id, start_time, duration, end_time in operations:
            job_key = f"J{job_id}"

            # Ensure job exists in problem data
            if job_key not in jobs:
                return False, f"Job {job_id} not found in problem data.", declared_makespan

            expected_operations = jobs[job_key]
            operation_index = job_next_operation_indices[job_key]

            # Ensure the operation is the next expected one for this job
            if operation_index >= len(expected_operations):
                return (
                    False,
                    f"Extra operation for Job {job_id} beyond expected operations.",
                    declared_makespan,
                )

            expected_machine_id, expected_duration = expected_operations[operation_index]
            if machine_id != expected_machine_id or duration != expected_duration:
                return (
                    False,
                    f"Operation for Job {job_id} does not match expected machine or duration. Expected Machine {expected_machine_id}, Duration {expected_duration}.",
                    declared_makespan,
                )

            # Validate operation duration and end time
            if end_time - start_time != duration:
                return (
                    False,
                    f"Incorrect duration for Job {job_id}, Machine {machine_id}.",
                    declared_makespan,
                )

            # Ensure that the operation does not start before the previous operation in the same job ends
            if job_id in job_end_times and start_time < job_end_times[job_id]:
                return (
                    False,
                    f"Job {job_id} starts operation at {start_time} before its previous operation ended at {job_end_times[job_id]}.",
                    declared_makespan,
                )

            # Ensure no overlapping jobs on the same machine (machine can only process one job at a time)
            if machine_id not in machine_operations:
                machine_operations[machine_id] = []
            machine_operations[machine_id].append((start_time, end_time))

            # Update machine's and job's last end times
            job_end_times[job_id] = end_time
            job_next_operation_indices[job_key] += 1

        # After collecting all operations, check for machine overlaps
        for machine_id, intervals in machine_operations.items():
            # Sort intervals by start time
            intervals.sort(key=lambda x: x[0])
            for i in range(1, len(intervals)):
                prev_end = intervals[i - 1][1]
                curr_start = intervals[i][0]
                if curr_start < prev_end:
                    return (
                        False,
                        f"Machine {machine_id} is overbooked. Overlap between times {prev_end} and {curr_start}.",
                        declared_makespan,
                    )

        # Ensure all operations for each job have been scheduled
        for job_key, expected_operations in jobs.items():
            if job_next_operation_indices[job_key] != len(expected_operations):
                return (
                    False,
                    f"Job {job_key} does not have all required operations scheduled.",
                    declared_makespan,
                )

    except Exception as e:
        return False, f"Error during validation: {str(e)}", declared_makespan

    # Calculate actual makespan
    actual_makespan = max(end_time for _, _, _, _, end_time in operations)

    # Adjusted to accept the solution and compute makespan if declared makespan is missing or incorrect
    if declared_makespan is None or actual_makespan != declared_makespan:
        message = f"Makespan was missing or incorrect. Computed makespan: {actual_makespan}"
        declared_makespan = actual_makespan
    else:
        message = f"Valid solution with correct makespan: {declared_makespan}"

    return True, message, declared_makespan

def parse_solution_list_of_lists(solution):
    """
    Parses a solution in list of lists format into operations.
    
    Args:
        solution (str): The solution text containing a list of lists.
        
    Returns:
        tuple: A tuple containing:
            - operations (list): List of operations in format (job_id, machine_id, start_time, duration, end_time)
            - declared_makespan (int or None): The makespan declared in the solution, or None if not found.
    """
    operations = []
    
    # Extract all operations using regex
    operation_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    matches = re.finditer(operation_pattern, solution)
    
    for match in matches:
        job_id = int(match.group(1))
        machine_id = int(match.group(2))
        start_time = int(match.group(3))
        duration = int(match.group(4))
        end_time = start_time + duration
        operations.append((job_id, machine_id, start_time, duration, end_time))
    
    # Extract makespan
    makespan_pattern = r'Maximum end completion time or Makespan: (\d+)'
    makespan_match = re.search(makespan_pattern, solution)
    
    if makespan_match:
        declared_makespan = int(makespan_match.group(1))
    else:
        declared_makespan = None
    
    return operations, declared_makespan

def validate_list_of_lists_format(problem_data, solution):
    """
    Validates a list of lists format solution against the problem data.
    
    Args:
        problem_data (str): The problem data.
        solution (str): The solution in list of lists format.
        
    Returns:
        tuple: A tuple containing:
            - valid (bool): Whether the solution is valid.
            - message (str): A message explaining the validation result.
            - makespan (int or None): The makespan of the solution, or None if invalid.
    """
    jobs = parse_problem_data(problem_data)

    if isinstance(jobs, tuple):  # Check if an error was encountered during parsing
        return False, f"Error parsing problem data: {jobs[1]}", None

    if not solution.strip():
        return False, "Solution string is empty.", None

    operations, declared_makespan = parse_solution_list_of_lists(solution)

    if not operations:
        return False, "No valid operations found in the solution.", declared_makespan

    # Ensure all jobs are accounted for
    solution_jobs = set(f"J{job_id}" for job_id, _, _, _, _ in operations)
    if solution_jobs != set(jobs.keys()):
        return (
            False,
            f"Missing jobs in the solution. Expected: {set(jobs.keys())}, Found: {solution_jobs}",
            declared_makespan,
        )

    # Create dictionaries to track the last end time for each machine and job
    machine_end_times = {}
    job_end_times = {}
    job_next_operation_indices = {job_id: 0 for job_id in jobs}

    # Prepare to track operations for each machine to detect overlaps
    machine_operations = {}

    try:
        # Sort operations by their start time to process them in chronological order
        operations.sort(key=lambda x: x[2])

        for job_id, machine_id, start_time, duration, end_time in operations:
            job_key = f"J{job_id}"

            # Ensure job exists in problem data
            if job_key not in jobs:
                return False, f"Job {job_id} not found in problem data.", declared_makespan

            expected_operations = jobs[job_key]
            operation_index = job_next_operation_indices[job_key]

            # Ensure the operation is the next expected one for this job
            if operation_index >= len(expected_operations):
                return (
                    False,
                    f"Extra operation for Job {job_id} beyond expected operations.",
                    declared_makespan,
                )

            expected_machine_id, expected_duration = expected_operations[operation_index]
            if machine_id != expected_machine_id or duration != expected_duration:
                return (
                    False,
                    f"Operation for Job {job_id} does not match expected machine or duration. Expected Machine {expected_machine_id}, Duration {expected_duration}.",
                    declared_makespan,
                )

            # Validate operation duration and end time
            if end_time - start_time != duration:
                return (
                    False,
                    f"Incorrect duration for Job {job_id}, Machine {machine_id}.",
                    declared_makespan,
                )

            # Ensure that the operation does not start before the previous operation in the same job ends
            if job_key in job_end_times and start_time < job_end_times[job_key]:
                return (
                    False,
                    f"Job {job_id} starts operation at {start_time} before its previous operation ended at {job_end_times[job_key]}.",
                    declared_makespan,
                )

            # Ensure no overlapping jobs on the same machine (machine can only process one job at a time)
            if machine_id not in machine_operations:
                machine_operations[machine_id] = []
            machine_operations[machine_id].append((start_time, end_time))

            # Update machine's and job's last end times
            job_end_times[job_key] = end_time
            job_next_operation_indices[job_key] += 1

        # After collecting all operations, check for machine overlaps
        for machine_id, intervals in machine_operations.items():
            # Sort intervals by start time
            intervals.sort(key=lambda x: x[0])
            for i in range(1, len(intervals)):
                prev_end = intervals[i - 1][1]
                curr_start = intervals[i][0]
                if curr_start < prev_end:
                    return (
                        False,
                        f"Machine {machine_id} is overbooked. Overlap between times {prev_end} and {curr_start}.",
                        declared_makespan,
                    )

        # Ensure all operations for each job have been scheduled
        for job_key, expected_operations in jobs.items():
            if job_next_operation_indices[job_key] != len(expected_operations):
                return (
                    False,
                    f"Job {job_key} does not have all required operations scheduled.",
                    declared_makespan,
                )

    except Exception as e:
        return False, f"Error during validation: {str(e)}", declared_makespan

    # Calculate actual makespan
    actual_makespan = max(end_time for _, _, _, _, end_time in operations)

    # Adjusted to accept the solution and compute makespan if declared makespan is missing or incorrect
    if declared_makespan is None or actual_makespan != declared_makespan:
        message = f"Makespan was missing or incorrect. Computed makespan: {actual_makespan}"
        declared_makespan = actual_makespan
    else:
        message = f"Valid solution with correct makespan: {declared_makespan}"

    return True, message, declared_makespan
