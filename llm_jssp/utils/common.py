import csv
import os


def get_last_processed_index(csv_filename):
    """Return last processed index in a CSV log; -1 if not found/empty."""
    if not os.path.exists(csv_filename):
        return -1

    try:
        with open(csv_filename, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            rows = list(reader)
            if len(rows) <= 1:
                return -1

            last_row = rows[-1]
            if last_row and last_row[0].isdigit():
                return int(last_row[0])
    except Exception:
        return -1

    return -1


def max_new_token_pred(num_jobs, num_machines):
    """Predict a conservative max_new_tokens budget from JSSP size."""
    base_tokens = num_jobs * num_machines * 30
    min_tokens = 3000
    max_tokens = 8000
    predicted_tokens = max(min_tokens, min(base_tokens, max_tokens))

    print(f"Jobs: {num_jobs}, Machines: {num_machines} -> Predicted tokens: {predicted_tokens}")
    return predicted_tokens
