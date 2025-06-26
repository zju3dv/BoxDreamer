#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

def load_json_file(path):
    """
    Load a JSON file and return the parsed data.

    Args:
        path (str): The file path to the JSON file.

    Returns:
        The parsed JSON data (expected to be a dict) or None if an error occurs.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def select_metrics(candidate_metrics):
    """
    Display candidate metrics (i.e. the JSON top-level keys) and
    allow the user to select which ones to convert.
    
    If no selection is made, all metrics will be used.

    Args:
        candidate_metrics (list): List of metric titles.
    
    Returns:
        A list of selected metrics.
    """
    print("Candidate Metrics:")
    for i, metric in enumerate(candidate_metrics, start=1):
        print(f"  {i}. {metric}")
    input_str = input("Enter the indices of metrics to convert (comma separated, default all): ").strip()
    if input_str == "":
        return candidate_metrics
    try:
        indices = [int(x.strip()) for x in input_str.split(",") if x.strip()]
    except ValueError:
        print("Invalid input; defaulting to all metrics.")
        return candidate_metrics

    selected = []
    for index in indices:
        if index < 1 or index > len(candidate_metrics):
            print(f"Index {index} is invalid and will be skipped.")
        else:
            selected.append(candidate_metrics[index - 1])
    if not selected:
        print("No valid metrics selected; defaulting to all metrics.")
        return candidate_metrics
    return selected

def choose_include_average():
    """
    Ask the user whether to include the average metric (with key "all").

    Returns:
        True if the user wants to include the "all" field; False otherwise.
    """
    ans = input("Include average metric (the 'all' field)? (Y/n): ").strip().lower()
    if ans == "n":
        return False
    return True

def select_objects(candidate_objects):
    """
    Display candidate objects and allow the user to select which ones to convert.
    Default is to include all objects.

    Args:
        candidate_objects (list): List of object IDs or names.
    
    Returns:
        A list of selected object keys.
    """
    print("Candidate Objects:")
    for i, obj in enumerate(candidate_objects, start=1):
        print(f"  {i}. {obj}")
    input_str = input("Enter the indices of objects to convert (comma separated, default all): ").strip()
    if input_str == "":
        return candidate_objects
    try:
        indices = [int(x.strip()) for x in input_str.split(",") if x.strip()]
    except ValueError:
        print("Invalid input; defaulting to all objects.")
        return candidate_objects

    selected = []
    for index in indices:
        if index < 1 or index > len(candidate_objects):
            print(f"Index {index} is invalid and will be skipped.")
        else:
            selected.append(candidate_objects[index - 1])
    if not selected:
        print("No valid objects selected; defaulting to all objects.")
        return candidate_objects
    return selected

def create_table_data(json_data, selected_metrics, selected_objects):
    """
    Create a list of dictionaries representing table data.
    Each row corresponds to one metric title. The first column is labeled "Metric",
    and subsequent columns correspond to the values for the selected objects.

    Additionally, this function asks the user:
      1. Whether to calculate the average value for each metric (i.e. a new "Average"
         column that is the mean of the selected object values).
      2. Whether to sort the table rows based on the computed "Average" value.

    Args:
        json_data (dict): The original JSON data.
        selected_metrics (list): The selected metric titles.
        selected_objects (list): The selected object keys (from the nested dictionaries).

    Returns:
        A list of dictionaries representing the table rows.
    """
    table_rows = []
    for metric in selected_metrics:
        row = {"Metric": metric}
        metric_values = json_data.get(metric, {})
        for obj in selected_objects:
            # Convert the value to a string; if missing, use an empty string.
            row[obj] = str(metric_values.get(obj, ""))
        table_rows.append(row)
    
    # Ask whether to calculate the average value for each metric
    calculate_avg = input("Calculate the average value for each metric? (Y/n): ").strip().lower()
    if calculate_avg == "y" or calculate_avg == "":
        # For each row, compute the mean of the selected object values.
        
        for row in table_rows:
            values = []
            for obj in selected_objects:
                val = row[obj]
                try:
                    v = float(val)
                    values.append(v)
                except ValueError:
                    continue  # Skip if the value is empty or not convertible to float.
            
            # add new column "Average" with the mean of the selected object values
            if values:
                row["Average"] = sum(values) / len(values)
            else:
                row["Average"] = ""
        
        if "Average" not in selected_objects:
            selected_objects.append("Average")
            
            
    
    return table_rows

def choose_template():
    """
    Provide LaTeX template choices and return the selected template option.

    Returns:
        A string indicating which template to use: "simple" or "booktabs".
    """
    print("\nChoose the LaTeX table template:")
    print("  1. Simple Table (basic tabular environment)")
    print("  2. Booktabs Table (formatted using the booktabs package)")
    print("  3. Markdown Table (for preview only)")
    choice = input("Enter the corresponding number (1 or 2): ").strip()
    if choice == "1":
        return "simple"
    elif choice == "2":
        return "booktabs"
    elif choice == "3":
        return "markdown"
    else:
        print("Invalid choice; defaulting to Simple Table.")
        return "simple"

def generate_simple_latex(rows, fields):
    """
    Generate LaTeX table code using a simple template with the tabular environment.
    Horizontal lines are generated using \\hline.

    Args:
        rows (list): List of dictionaries (table rows).
        fields (list): List of field names (column headers).

    Returns:
        A string containing the LaTeX table code.
    """
    col_spec = "|" + "c|" * len(fields)
    header = " & ".join(fields) + " \\\\ \\hline"
    body = ""
    for row in rows:
        row_values = [row.get(field, "") for field in fields]
        body += " & ".join(row_values) + " \\\\ \\hline\n"
    latex_table = (
        "\\begin{tabular}{" + col_spec + "}\n"
        "\\hline\n" +
        header + "\n" +
        body +
        "\\end{tabular}"
    )
    return latex_table

def generate_booktabs_latex(rows, fields):
    """
    Generate LaTeX table code using the booktabs template.
    This style uses the booktabs package commands like \\toprule, \\midrule, and \\bottomrule.

    Args:
        rows (list): List of dictionaries (table rows).
        fields (list): List of field names (column headers).

    Returns:
        A string containing the LaTeX table code.
    """
    # Here we set the first column to left alignment and others to center.
    if fields:
        col_spec = "l" + "c" * (len(fields) - 1)
    else:
        col_spec = ""
    header = " & ".join(fields) + " \\\\"
    body = ""
    for row in rows:
        row_values = [row.get(field, "") for field in fields]
        body += " & ".join(row_values) + " \\\\ \n"
    latex_table = (
        "\\begin{tabular}{" + col_spec + "}\n"
        "\\toprule\n" +
        header + "\n" +
        "\\midrule\n" +
        body +
        "\\bottomrule\n"
        "\\end{tabular}"
    )
    return latex_table

def generate_markdown_table(rows, fields):
    """
    Generate a Markdown table preview based on the table data.
    This is useful for a quick visual inspection of the output.

    Args:
        rows (list): List of dictionaries representing table rows.
        fields (list): List of field names (column headers).

    Returns:
        A string containing the Markdown table.
    """
    lines = []
    # Build header row.
    header_cells = [" " + str(cell) + " " for cell in fields]
    lines.append("|" + "|".join(header_cells) + "|")
    # Build separator row.
    separator_cells = [":" + "-" * (len(str(cell)) - 2) + ":" for cell in fields]
    lines.append("|" + "|".join(separator_cells) + "|")
    # Build body rows.
    for row in rows:
        row_cells = [" " + str(row.get(field, "")) + " " for field in fields]
        lines.append("|" + "|".join(row_cells) + "|")
    return "\n".join(lines)

def generate_ascii_table(rows, fields):
    """
    Generate an ASCII table preview based on the table data.
    This is useful for a quick visual inspection of the output.

    Args:
        rows (list): List of dictionaries representing table rows.
        fields (list): List of field names (column headers).

    Returns:
        A string containing the ASCII table.
    """
    # Build a 2D list (header + rows).
    table_data = [fields]
    for row in rows:
        row_line = [row.get(field, "") for field in fields]
        table_data.append(row_line)

    # Compute the maximum width for each column.
    col_widths = []
    for col_idx in range(len(fields)):
        max_width = max(len(str(row[col_idx])) for row in table_data)
        col_widths.append(max_width)
    horizontal_line = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    lines = [horizontal_line]
    # Build header row.
    header_cells = []
    for i, cell in enumerate(table_data[0]):
        header_cells.append(" " + str(cell).ljust(col_widths[i]) + " ")
    lines.append("|" + "|".join(header_cells) + "|")
    lines.append(horizontal_line)
    # Build body rows.
    for row in table_data[1:]:
        row_cells = []
        for i, cell in enumerate(row):
            row_cells.append(" " + str(cell).ljust(col_widths[i]) + " ")
        lines.append("|" + "|".join(row_cells) + "|")
    lines.append(horizontal_line)
    return "\n".join(lines)

def main():
    """
    Main function to run the JSON to LaTeX table conversion tool.
    """
    print("=== JSON to LaTeX Table Converter ===\n")
    
    # 1. Input JSON file path.
    json_path = input("Enter the JSON file path: ").strip()
    if not os.path.exists(json_path):
        print("File does not exist.")
        return

    json_data = load_json_file(json_path)
    if json_data is None:
        return

    # Ensure JSON data is a dictionary.
    if not isinstance(json_data, dict):
        print("The JSON data must be a dictionary (with metric titles as keys).")
        return

    # 2. Let the user select which metrics to convert.
    candidate_metrics = list(json_data.keys())
    if not candidate_metrics:
        print("No metrics found in the JSON file.")
        return
    selected_metrics = select_metrics(candidate_metrics)
    
    # 3. Extract candidate objects from one metric (assumed consistent across metrics).
    # Use the first selected metric.
    sample_metric = selected_metrics[0]
    if not isinstance(json_data[sample_metric], dict):
        print(f"Expected the value for metric '{sample_metric}' to be a dictionary.")
        return
    candidate_objects = list(json_data[sample_metric].keys())
    if not candidate_objects:
        print("No object data found in the metric dictionaries.")
        return

    # 4. Ask the user whether to include the average value (i.e. the 'all' field).
    include_avg = choose_include_average()
    if not include_avg:
        candidate_objects = [obj for obj in candidate_objects if obj.lower() != "all"]

    # 5. Optionally allow the user to select a subset of objects.
    selected_objects = select_objects(candidate_objects)

    # 6. Create table data.
    # The output table will have a first column "Metric" and subsequent columns are the selected object keys.
    table_data = create_table_data(json_data, selected_metrics, selected_objects)
    columns = ["Metric"] + selected_objects

    # 7. Let the user choose a LaTeX table template.
    template = choose_template()

    # 8. Generate LaTeX code using the chosen template.
    if template == "simple":
        latex_code = generate_simple_latex(table_data, columns)
    elif template == "booktabs":
        latex_code = generate_booktabs_latex(table_data, columns)
    elif template == "markdown":
        markdown_table = generate_markdown_table(table_data, columns)
        print("\nGenerated Markdown table code:\n")
        print(markdown_table)
        return
    else:
        latex_code = "Unknown template option."

    # 9. Output LaTeX code.
    print("\nGenerated LaTeX table code:\n")
    print(latex_code)

    # 10. Offer an ASCII preview.
    preview = input("\nWould you like to preview the table in ASCII? (y/n): ").strip().lower()
    if preview == "y":
        ascii_table = generate_ascii_table(table_data, columns)
        print("\nTable Preview (ASCII):\n")
        print(ascii_table)

    # 11. Optionally save the generated LaTeX code to a file.
    save_option = input("\nWould you like to save the LaTeX code to a file? (y/n): ").strip().lower()
    if save_option == "y":
        output_path = input("Enter the output file name (e.g., table.tex): ").strip()
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(latex_code)
            print(f"LaTeX code saved to {output_path}")
        except Exception as e:
            print("Error saving file:", e)

if __name__ == "__main__":
    main()