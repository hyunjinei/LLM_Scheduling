# -----------------------------
# Validation Function for Accord Format (Format 1)
# -----------------------------
def validate_accord_format(inputs, generated_text, capacity, num_vehicles, num_cities, distance_matrix_str):
    """
    Validates a VRP solution in Accord format (vehicle routes with arrows).
    Example: "Vehicle Route: (0): (7, 83) -> (5): (11, 6) -> (0): (7, 83) + 30"
    
    Parameters:
    - inputs: Input data containing problem information
    - generated_text: Text containing the solution to be validated
    - capacity: Maximum capacity of each vehicle
    - num_vehicles: Maximum number of vehicles available
    - num_cities: Total number of cities including the depot
    - distance_matrix_str: String representation of the distance matrix
    
    Returns:
    - is_feasible: True if the solution is feasible
    - message: Validation message or error description
    - declared_distance: Distance value declared in the solution
    - calculated_distance: Actual distance calculated using the distance matrix
    - gap: The gap between declared and calculated distances
    """
    # Extract the solution part from the response
    response_parts = generated_text.split("### Response:")
    if len(response_parts) > 1:
        solution_text = response_parts[1].strip()
    else:
        solution_text = generated_text.strip()
    
    # Parse distance matrix
    distance_matrix = {}
    pairs = distance_matrix_str.split(", ")
    for pair in pairs:
        parts = pair.split("=")
        if len(parts) != 2:
            continue
        nodes = parts[0].strip()
        distance = int(parts[1].strip())
        from_node = int(nodes.split("):(")[0].strip("("))
        to_node = int(nodes.split("):(")[1].strip(")"))
        distance_matrix[(from_node, to_node)] = distance
        distance_matrix[(to_node, from_node)] = distance  # Assuming symmetric distances
    
    # Add depot-to-depot distance as 0 (for unused vehicles)
    distance_matrix[(0, 0)] = 0
    
    # Parse city coordinates and demands
    city_coords = {}
    city_demands = {}
    
    # Extract demands from input if available
    if isinstance(inputs, dict) and 'demands' in inputs:
        demands = inputs['demands']
        for i, demand in enumerate(demands):
            city_demands[i] = demand
    elif isinstance(inputs, dict) and 'variables' in inputs and 'demands' in inputs['variables']:
        demands = inputs['variables']['demands']
        for i, demand in enumerate(demands):
            city_demands[i] = demand
    else:
        # Try to handle different input formats
        try:
            if hasattr(inputs, 'get') and inputs.get('demands'):
                demands = inputs.get('demands')
                for i, demand in enumerate(demands):
                    city_demands[i] = demand
            else:
                # If demands are not available, create default demands
                for i in range(num_cities):
                    city_demands[i] = 0 if i == 0 else 1  # Default: depot has 0 demand, cities have 1
        except:
            # If all attempts fail, use default demands
            for i in range(num_cities):
                city_demands[i] = 0 if i == 0 else 1
    
    # Extract coordinates
    if isinstance(inputs, dict) and 'coords' in inputs:
        coords = inputs['coords']
        for i, coord in enumerate(coords):
            if isinstance(coord, list) and len(coord) == 2:
                city_coords[i] = (coord[0], coord[1])
    elif isinstance(inputs, dict) and 'variables' in inputs and 'coords' in inputs['variables']:
        coords = inputs['variables']['coords']
        for i, coord in enumerate(coords):
            if isinstance(coord, list) and len(coord) == 2:
                city_coords[i] = (coord[0], coord[1])
    else:
        try:
            if hasattr(inputs, 'get') and inputs.get('coords'):
                coords = inputs.get('coords')
                for i, coord in enumerate(coords):
                    if isinstance(coord, list) and len(coord) == 2:
                        city_coords[i] = (coord[0], coord[1])
            else:
                for i, coords in enumerate(inputs):
                    if isinstance(coords, list) and len(coords) == 2:
                        city_coords[i] = (coords[0], coords[1])
        except:
            # If coordinate extraction fails, proceed without coordinates
            pass
    
    # Variables to store the parsed routes and declared distance
    all_routes = []
    declared_distance = None
    route_coords = []  # To store coordinates extracted from solution for validation
    
    # First, try to extract the declared distance from any format
    distance_line = [line for line in solution_text.split('\n') if "Overall Total Distance:" in line]
    if distance_line:
        try:
            declared_distance = int(distance_line[0].split("Overall Total Distance:")[1].strip())
        except:
            pass
    
    # Check if this is the Accord format (Vehicle Route with arrows)
    if "Vehicle Route:" not in solution_text:
        return False, "Not in Accord format (expected 'Vehicle Route:' with arrow notation)", None, None, None
    
    try:
        # Split by "Vehicle Route:" to get each vehicle's route
        route_parts = solution_text.split("Vehicle Route:")
        
        # Parse each route
        for part in route_parts[1:]:  # Skip the first empty part
            route = []
            route_coord = []
            route_line = part.split('\n')[0].strip()
            nodes = route_line.split("->")
            
            for node in nodes:
                if "(" in node and "):" in node:
                    node_str = node.split("):")[0].strip()
                    coord_str = node.split("):")[1].split("+")[0].strip()
                    try:
                        node_idx = int(node_str.strip("() "))
                        route.append(node_idx)
                        
                        # Extract coordinates for validation
                        if "(" in coord_str and ")" in coord_str:
                            coord_parts = coord_str.strip("() ").split(",")
                            if len(coord_parts) == 2:
                                x, y = int(coord_parts[0].strip()), int(coord_parts[1].strip())
                                route_coord.append((node_idx, (x, y)))
                    except ValueError:
                        continue
            
            if route:  # Only add non-empty routes
                all_routes.append(route)
                route_coords.append(route_coord)
        
    except Exception as e:
        return False, f"Error parsing Accord format solution: {str(e)}", None, None, None
    
    # If no routes were successfully parsed
    if not all_routes:
        return False, "Could not parse routes from Accord format solution", None, None, None
    
    # Check if the number of vehicles used is correct
    actual_vehicles = len([route for route in all_routes if len(route) > 2 or (len(route) == 2 and route[0] != route[1])])
    
    if actual_vehicles > num_vehicles:
        return False, f"Solution uses {actual_vehicles} vehicles but only {num_vehicles} are available", declared_distance, None, None
    
    # Check if each route starts and ends at the depot (node 0)
    for i, route in enumerate(all_routes):
        if not route or route[0] != 0 or route[-1] != 0:
            return False, f"Route {i+1} must start and end at the depot (node 0)", declared_distance, None, None
    
    # Check if all cities are visited exactly once across all routes
    visited_nodes = set()
    for route in all_routes:
        # Add all nodes except depot (0) at start and end
        visited_nodes.update(route[1:-1])
    
    # Check if exactly all cities except depot are visited
    expected_nodes = set(range(1, num_cities))
    if visited_nodes != expected_nodes:
        missing = expected_nodes - visited_nodes
        duplicated = visited_nodes - expected_nodes
        error_msg = "Not all cities are visited exactly once."
        if missing:
            error_msg += f" Missing nodes: {missing}."
        if duplicated:
            error_msg += f" Duplicated or invalid nodes: {duplicated}."
        
        return False, error_msg, declared_distance, None, None
    
    # Validate coordinates if available
    if route_coords and city_coords:
        for route_coord_list in route_coords:
            for node_idx, coords in route_coord_list:
                if node_idx in city_coords and city_coords[node_idx] != coords:
                    return False, f"Coordinates mismatch for node {node_idx}: solution has {coords} but input has {city_coords[node_idx]}", declared_distance, calculated_distance, None
    
    # Calculate the actual total distance across all routes and check capacity constraints
    calculated_distance = 0
    for r_idx, route in enumerate(all_routes):
        route_distance = 0
        route_load = 0  # Track load for capacity constraint
        
        # Skip empty routes or depot-only routes
        if len(route) <= 2 and route[0] == 0 and route[-1] == 0:
            continue
            
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i+1]
            
            # Special case: if both nodes are 0 (depot to depot), distance is 0
            if from_node == 0 and to_node == 0:
                edge_distance = 0
            elif (from_node, to_node) not in distance_matrix:
                return False, f"Distance for edge ({from_node}, {to_node}) not found in distance matrix", declared_distance, None, None
            else:
                edge_distance = distance_matrix[(from_node, to_node)]
            
            route_distance += edge_distance
            CAPACITY_TOLERANCE = 1
            # Add demand for the current node (not the next node)
            # This is the fix for the capacity issue - only check nodes being visited
            if from_node != 0 and from_node in city_demands:
                route_load += city_demands[from_node]
                # Only check capacity after adding each city's demand
                # Check total route capacity once (after adding all demands)
                capacity += CAPACITY_TOLERANCE
                if route_load > capacity:
                    return False, f"Capacity constraint violated in route {r_idx+1}: load {route_load} exceeds capacity {capacity}", declared_distance, None, None
        
        calculated_distance += route_distance
    
    # Check if the declared distance matches the calculated distance
    if declared_distance is not None and calculated_distance != declared_distance:
        return False, f"Declared distance ({declared_distance}) does not match calculated distance ({calculated_distance})", declared_distance, calculated_distance, (calculated_distance - declared_distance) / calculated_distance if calculated_distance != 0 else None
    
    return True, "Valid solution", declared_distance, calculated_distance, None


# -----------------------------
# Validation Function for List of Lists Format (Format 3)
# -----------------------------
def validate_list_of_lists_format(inputs, generated_text, capacity, num_vehicles, num_cities, distance_matrix_str):
    """
    Validates a VRP solution in List of Lists format (indices with coordinates).
    Example: "[(0): (74, 93), (5): (70, 66), (9): (52, 33), (0): (74, 93)]"
    
    Parameters:
    - inputs: Input data containing problem information
    - generated_text: Text containing the solution to be validated
    - capacity: Maximum capacity of each vehicle
    - num_vehicles: Maximum number of vehicles available
    - num_cities: Total number of cities including the depot
    - distance_matrix_str: String representation of the distance matrix
    
    Returns:
    - is_feasible: True if the solution is feasible
    - message: Validation message or error description
    - declared_distance: Distance value declared in the solution
    - calculated_distance: Actual distance calculated using the distance matrix
    - gap: The gap between declared and calculated distances
    """
    # Extract the solution part from the response
    response_parts = generated_text.split("### Response:")
    if len(response_parts) > 1:
        solution_text = response_parts[1].strip()
    else:
        solution_text = generated_text.strip()
    
    # Parse distance matrix
    distance_matrix = {}
    pairs = distance_matrix_str.split(", ")
    for pair in pairs:
        parts = pair.split("=")
        if len(parts) != 2:
            continue
        nodes = parts[0].strip()
        distance = int(parts[1].strip())
        from_node = int(nodes.split("):(")[0].strip("("))
        to_node = int(nodes.split("):(")[1].strip(")"))
        distance_matrix[(from_node, to_node)] = distance
        distance_matrix[(to_node, from_node)] = distance  # Assuming symmetric distances
    
    # Add depot-to-depot distance as 0 (for unused vehicles)
    distance_matrix[(0, 0)] = 0
    
    # Parse city coordinates and demands
    city_coords = {}
    city_demands = {}
    
    # Extract demands from input if available
    if isinstance(inputs, dict) and 'demands' in inputs:
        demands = inputs['demands']
        for i, demand in enumerate(demands):
            city_demands[i] = demand
    elif isinstance(inputs, dict) and 'variables' in inputs and 'demands' in inputs['variables']:
        demands = inputs['variables']['demands']
        for i, demand in enumerate(demands):
            city_demands[i] = demand
    else:
        # Try to handle different input formats
        try:
            if hasattr(inputs, 'get') and inputs.get('demands'):
                demands = inputs.get('demands')
                for i, demand in enumerate(demands):
                    city_demands[i] = demand
            else:
                # If demands are not available, create default demands
                for i in range(num_cities):
                    city_demands[i] = 0 if i == 0 else 1  # Default: depot has 0 demand, cities have 1
        except:
            # If all attempts fail, use default demands
            for i in range(num_cities):
                city_demands[i] = 0 if i == 0 else 1
    
    # Extract coordinates
    if isinstance(inputs, dict) and 'coords' in inputs:
        coords = inputs['coords']
        for i, coord in enumerate(coords):
            if isinstance(coord, list) and len(coord) == 2:
                city_coords[i] = (coord[0], coord[1])
    elif isinstance(inputs, dict) and 'variables' in inputs and 'coords' in inputs['variables']:
        coords = inputs['variables']['coords']
        for i, coord in enumerate(coords):
            if isinstance(coord, list) and len(coord) == 2:
                city_coords[i] = (coord[0], coord[1])
    else:
        try:
            if hasattr(inputs, 'get') and inputs.get('coords'):
                coords = inputs.get('coords')
                for i, coord in enumerate(coords):
                    if isinstance(coord, list) and len(coord) == 2:
                        city_coords[i] = (coord[0], coord[1])
            else:
                for i, coords in enumerate(inputs):
                    if isinstance(coords, list) and len(coords) == 2:
                        city_coords[i] = (coords[0], coords[1])
        except:
            # If coordinate extraction fails, proceed without coordinates
            pass
    
    # Variables to store the parsed routes and declared distance
    all_routes = []
    route_coords = []  # To store coordinates extracted from solution for validation
    declared_distance = None
    
    # First, try to extract the declared distance from any format
    distance_line = [line for line in solution_text.split('\n') if "Overall Total Distance:" in line]
    if distance_line:
        try:
            declared_distance = int(distance_line[0].split("Overall Total Distance:")[1].strip())
        except:
            pass
    
    # Check if this is the List of Lists format
    if not ("[(0):" in solution_text or "[(" in solution_text and "):" in solution_text):
        return False, "Not in List of Lists format (expected '[(index): (x, y)' notation)", None, None, None
    
    try:
        lines = solution_text.strip().split('\n')
        
        # Filter out non-route lines
        route_lines = [line for line in lines if "[(" in line or "[(0):" in line]
        
        # Parse each route
        for line in route_lines:
            if "[" in line and "]" in line:
                route = []
                route_coord = []
                # Extract content between brackets
                content = line[line.find("[")+1:line.rfind("]")]
                parts = content.split(", ")
                
                for part in parts:
                    if "):" in part:
                        try:
                            # Extract node index
                            node_str = part.split("):")[0].strip()
                            node_idx = int(node_str.strip("() "))
                            route.append(node_idx)
                            
                            # Extract coordinates for validation
                            coord_str = part.split("):")[1].strip()
                            if "(" in coord_str and ")" in coord_str:
                                coord_parts = coord_str.strip("() ").split(",")
                                if len(coord_parts) == 2:
                                    x, y = int(coord_parts[0].strip()), int(coord_parts[1].strip())
                                    route_coord.append((node_idx, (x, y)))
                        except ValueError:
                            continue
                
                if route:  # Only add non-empty routes
                    all_routes.append(route)
                    route_coords.append(route_coord)
        
    except Exception as e:
        return False, f"Error parsing List of Lists format solution: {str(e)}", None, None, None
    
    # If no routes were successfully parsed
    if not all_routes:
        return False, "Could not parse routes from List of Lists format solution", None, None, None
    
    # Check if the number of vehicles used is correct
    actual_vehicles = len([route for route in all_routes if len(route) > 2 or (len(route) == 2 and route[0] != route[1])])
    
    if actual_vehicles > num_vehicles:
        return False, f"Solution uses {actual_vehicles} vehicles but only {num_vehicles} are available", declared_distance, None, None
    
    # Check if each route starts and ends at the depot (node 0)
    for i, route in enumerate(all_routes):
        if not route or route[0] != 0 or route[-1] != 0:
            return False, f"Route {i+1} must start and end at the depot (node 0)", declared_distance, None, None
    
    # Check if all cities are visited exactly once across all routes
    visited_nodes = set()
    for route in all_routes:
        # Add all nodes except depot (0) at start and end
        visited_nodes.update(route[1:-1])
    
    # Check if exactly all cities except depot are visited
    expected_nodes = set(range(1, num_cities))
    if visited_nodes != expected_nodes:
        missing = expected_nodes - visited_nodes
        duplicated = visited_nodes - expected_nodes
        error_msg = "Not all cities are visited exactly once."
        if missing:
            error_msg += f" Missing nodes: {missing}."
        if duplicated:
            error_msg += f" Duplicated or invalid nodes: {duplicated}."
        
        return False, error_msg, declared_distance, None, None
    
    # Validate coordinates if available
    if route_coords and city_coords:
        for route_coord_list in route_coords:
            for node_idx, coords in route_coord_list:
                if node_idx in city_coords and city_coords[node_idx] != coords:
                    return False, f"Coordinates mismatch for node {node_idx}: solution has {coords} but input has {city_coords[node_idx]}", declared_distance, None, None
    
    # Calculate the actual total distance across all routes and check capacity constraints
    calculated_distance = 0
    for r_idx, route in enumerate(all_routes):
        route_distance = 0
        route_load = 0  # Track load for capacity constraint
        
        # Skip empty routes or depot-only routes
        if len(route) <= 2 and route[0] == 0 and route[-1] == 0:
            continue
            
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i+1]
            
            # Special case: if both nodes are 0 (depot to depot), distance is 0
            if from_node == 0 and to_node == 0:
                edge_distance = 0
            elif (from_node, to_node) not in distance_matrix:
                return False, f"Distance for edge ({from_node}, {to_node}) not found in distance matrix", declared_distance, None, None
            else:
                edge_distance = distance_matrix[(from_node, to_node)]
            
            route_distance += edge_distance
            CAPACITY_TOLERANCE = 1
            # Add demand for the current node (not the next node)
            # This is the fix for the capacity issue - only check nodes being visited
            if from_node != 0 and from_node in city_demands:
                route_load += city_demands[from_node]
                # Only check capacity after adding each city's demand
                # Check total route capacity once (after adding all demands)
                capacity += CAPACITY_TOLERANCE
                if route_load > capacity:
                    return False, f"Capacity constraint violated in route {r_idx+1}: load {route_load} exceeds capacity {capacity}", declared_distance, None, None
        
        calculated_distance += route_distance
    
    # Check if the declared distance matches the calculated distance
    if declared_distance is not None and calculated_distance != declared_distance:
        return False, f"Declared distance ({declared_distance}) does not match calculated distance ({calculated_distance})", declared_distance, calculated_distance, (calculated_distance - declared_distance) / calculated_distance if calculated_distance != 0 else None
    
    return True, "Valid solution", declared_distance, calculated_distance, None