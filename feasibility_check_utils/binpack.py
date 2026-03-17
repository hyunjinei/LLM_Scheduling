import re
import ast

def validate_list_of_lists_format(inputs, generated_text, bin_capacity, num_bins_sol):
    """
    Validates a bin packing solution in list of lists format.
    
    Args:
        inputs: String representation of items as [(id, weight), ...]
        generated_text: Generated solution text containing a list of lists
        bin_capacity: Maximum capacity of each bin
        num_bins_sol: Optimal number of bins (for reference)
        
    Returns:
        Tuple of (is_feasible, message, computed_value, num_bins_sol)
    """
    # Parse input items
    try:
        items = ast.literal_eval(inputs)
        item_weights = {item_id: weight for item_id, weight in items}
        num_items = len(items)  # Extract num_items from inputs
    except Exception as e:
        return False, f"Failed to parse input items: {str(e)}", None, num_bins_sol
    
    # Try different methods to extract the solution
    bin_assignments = None
    
    # Method 1: Try to find a Python list pattern
    try:
        # Clean up the text to help with parsing - remove line breaks and extraneous spaces
        clean_text = re.sub(r'\s+', ' ', generated_text)
        
        # Look for list of lists patterns
        list_patterns = [
            r'\[\s*\[.*?\]\s*\]',  # Standard list of lists
            r'\[\s*\[[\d\s,]+\](?:\s*,\s*\[[\d\s,]+\])*\s*\]',  # More specific format
            r'\[\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\](?:\s*,\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\])*\s*\]'  # Very specific
        ]
        
        for pattern in list_patterns:
            match = re.search(pattern, clean_text, re.DOTALL)
            if match:
                bin_assignments_str = match.group(0)
                bin_assignments = ast.literal_eval(bin_assignments_str)
                break
    except Exception as e:
        # Continue to next method if this fails
        pass
    
    # Method 2: Try to extract lists line by line
    if bin_assignments is None:
        try:
            lines = generated_text.split('\n')
            potential_lists = []
            for line in lines:
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    try:
                        bin_list = ast.literal_eval(line)
                        if isinstance(bin_list, list) and all(isinstance(item, int) for item in bin_list):
                            potential_lists.append(bin_list)
                    except:
                        pass
            
            if potential_lists:
                bin_assignments = potential_lists
        except Exception as e:
            pass
    
    # Method 3: Look for bin assignments in text format
    if bin_assignments is None:
        try:
            bin_pattern = r'Bin\s+\d+\s*:\s*(\[[\d\s,]+\])'
            bin_matches = re.finditer(bin_pattern, generated_text)
            potential_lists = []
            
            for match in bin_matches:
                try:
                    bin_list = ast.literal_eval(match.group(1))
                    if isinstance(bin_list, list) and all(isinstance(item, int) for item in bin_list):
                        potential_lists.append(bin_list)
                except:
                    pass
            
            if potential_lists:
                bin_assignments = potential_lists
        except Exception as e:
            pass
    
    # If we still can't find a valid bin assignment, return error
    if bin_assignments is None:
        return False, "Could not extract a valid bin assignment list from the output", None, num_bins_sol
    
    # If the generated solution is a flat list, check if it's meant to be a single bin
    if bin_assignments and isinstance(bin_assignments[0], int):
        # If the total items in this list equals num_items, it's likely a single bin solution
        if len(bin_assignments) == num_items:
            bin_assignments = [bin_assignments]
        else:
            # It might be a list of bin indices, not a valid solution
            return False, "Found a list of integers but not a valid bin assignment", None, num_bins_sol
    
    # Count bins and validate
    computed_value = len(bin_assignments)
    
    # Check if all items are assigned exactly once
    all_assigned_items = [item for bin_items in bin_assignments for item in bin_items]
    
    if len(all_assigned_items) != num_items:
        return False, f"Solution assigns {len(all_assigned_items)} items, but there are {num_items} items", computed_value, num_bins_sol
    
    if len(all_assigned_items) != len(set(all_assigned_items)):
        duplicate_items = [item for item in set(all_assigned_items) if all_assigned_items.count(item) > 1]
        return False, f"Items {duplicate_items} are assigned to multiple bins", computed_value, num_bins_sol
    
    expected_items = set(range(num_items))
    if set(all_assigned_items) != expected_items:
        missing_items = expected_items - set(all_assigned_items)
        extra_items = set(all_assigned_items) - expected_items
        error_msg = ""
        if missing_items:
            error_msg += f"Missing items: {missing_items}. "
        if extra_items:
            error_msg += f"Unexpected items: {extra_items}. "
        return False, error_msg, computed_value, num_bins_sol
    
    # Check bin capacity constraints
    for bin_idx, bin_items in enumerate(bin_assignments):
        try:
            bin_weight = sum(item_weights[item_id] for item_id in bin_items)
            if bin_weight > bin_capacity:
                return False, f"Bin {bin_idx+1} exceeds capacity: {bin_weight} > {bin_capacity}", computed_value, num_bins_sol
        except KeyError as e:
            return False, f"Invalid item ID in bin {bin_idx+1}: {str(e)}", computed_value, num_bins_sol
    
    # Calculate gap between solution and optimal
    gap = (computed_value - num_bins_sol) / num_bins_sol if num_bins_sol > 0 else 0
    gap_percentage = gap * 100
    
    if gap <= 0:
        gap_message = f"Optimal solution found! Using {computed_value} bins."
    else:
        gap_message = f"Solution uses {computed_value} bins, which is {gap_percentage:.2f}% more than the optimal {num_bins_sol} bins."
    
    return True, gap_message, computed_value, num_bins_sol




import re
import ast

def validate_accord_format(inputs, generated_text, bin_capacity, num_bins_sol):
    """
    Validates a bin packing solution in accord format.
    
    Args:
        inputs: String representation of items as [(id, weight), ...]
        generated_text: Generated solution text in accord format
        bin_capacity: Maximum capacity of each bin
        num_bins_sol: Optimal number of bins (for reference)
        
    Returns:
        Tuple of (is_feasible, message, computed_value, num_bins_sol)
    """
    # Parse input items
    try:
        items = ast.literal_eval(inputs)
        item_weights = {item_id: weight for item_id, weight in items}
        num_items = len(items)  # Extract num_items from inputs
    except Exception as e:
        return False, f"Failed to parse input items: {str(e)}", None, num_bins_sol
    
    # Parse the generated text to extract bin assignments
    bin_assignments = []  # Will store lists of item IDs for each bin
    bin_loads = []        # Will store total weight of each bin
    
    # Try to detect the declared number of bins
    declared_value = None
    total_bins_pattern = r'Total bins required:\s*(\d+)'
    total_bins_match = re.search(total_bins_pattern, generated_text)
    if total_bins_match:
        declared_value = int(total_bins_match.group(1))
    
    # Process each bin from the text
    # Regex to match bin assignments: Bin X: (item_id, weight) -> ... <= bin_capacity
    bin_pattern = r'Bin\s+(\d+):\s+((?:\(\s*\d+\s*,\s*\d+\s*\)\s*->.*?)+)\s*<=\s*\d+'
    bin_matches = re.finditer(bin_pattern, generated_text, re.DOTALL)
    
    all_assigned_items = set()
    
    for bin_match in bin_matches:
        bin_number = int(bin_match.group(1))
        bin_content = bin_match.group(2)
        
        # Extract item IDs and weights
        item_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
        item_matches = re.finditer(item_pattern, bin_content)
        
        bin_items = []
        bin_weight_calculated = 0  # Our calculated weight
        
        for item_match in item_matches:
            item_id = int(item_match.group(1))
            # We don't use the weight from the text, we use our own dictionary
            bin_items.append(item_id)
            
            # Check for duplicates
            if item_id in all_assigned_items:
                return False, f"Item {item_id} is assigned to multiple bins", None, num_bins_sol
            
            all_assigned_items.add(item_id)
            
            # Calculate weight based on our item_weights dictionary
            if item_id in item_weights:
                bin_weight_calculated += item_weights[item_id]
            else:
                return False, f"Invalid item ID {item_id} in bin {bin_number}", None, num_bins_sol
        
        # Add to our collections
        bin_assignments.append(bin_items)
        bin_loads.append(bin_weight_calculated)
        
        # Check bin capacity constraint
        if bin_weight_calculated > bin_capacity:
            return False, f"Bin {bin_number} exceeds capacity: {bin_weight_calculated} > {bin_capacity}", len(bin_assignments), num_bins_sol
    
    # If we couldn't extract bins or the number doesn't match what was declared
    computed_value = len(bin_assignments)
    if computed_value == 0:
        return False, "Could not extract any valid bin assignments from the output", None, num_bins_sol
    
    # Check if all items are assigned
    expected_items = set(range(num_items))
    if all_assigned_items != expected_items:
        missing_items = expected_items - all_assigned_items
        extra_items = all_assigned_items - expected_items
        error_msg = ""
        if missing_items:
            error_msg += f"Missing items: {missing_items}. "
        if extra_items:
            error_msg += f"Unexpected items: {extra_items}. "
        return False, error_msg, computed_value, num_bins_sol
    
    # Calculate gap between solution and optimal
    gap = (computed_value - num_bins_sol) / num_bins_sol if num_bins_sol > 0 else 0
    gap_percentage = gap * 100
    
    if gap <= 0:
        gap_message = f"Optimal solution found! Using {computed_value} bins."
    else:
        gap_message = f"Solution uses {computed_value} bins, which is {gap_percentage:.2f}% more than the optimal {num_bins_sol} bins."
    
    return True, gap_message, computed_value, num_bins_sol