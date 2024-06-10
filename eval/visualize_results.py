# import json
# import pandas as pd
# from tabulate import tabulate

# IGNORE_MODELS = ["llama3_exp4_1", "llama3_exp4"]

# model_name_map = {
#     "llama3_exp1": "Fluency ❌\n Culture ❌\n Diversity ❌",
#     "llama3_exp3": "Fluency ✅\n Culture ❌\n Diversity ❌",
#     "llama3_exp2": "Fluency ❌\n Culture ✅\n Diversity ❌",
#     "llama3_exp4": "Fluency ❌\n Culture ❌\n Diversity ✅\n(Ultrachat)",
#     "llama3_exp4_1": "Fluency ❌\n Culture ❌\n Diversity ✅\n(Ultrachat Reduced)",
#     "llama3_exp4_2": "Fluency ❌\n Culture ❌\n Diversity ✅",
#     "llama3_clsit": "Fluency ✅\n Culture ✅\n Diversity ✅",
#     "llama3-wangchanx-demo": "WangchanX Llama3 8B",
#     "llama3-typhoon-v1.5-8b": "Typhoon-v1.5 8B",
#     "openthaigpt-v1-7b": "OpenThai 1.0.0 7B",
#     "llama3_abl2": "Self-Instruct (Translated)"
# }

# variant_name_map = {
#     "yes": "\n(Thai Culture Test Set)",
#     "no": "\n(General Test Set)"
# }

# # Load the JSON data from the file
# with open("evaluation_results.json", "r") as file:
#     data = json.load(file)

# # Sort the keys in data by the order in which they appear in model_name_map
# data = {k: data[k] for k in sorted(data, key=lambda x: list(model_name_map.keys()).index(x))}

# # Extract the tasks, models, and metrics
# tasks = set()
# models = []
# metrics = set()
# for model_name, model_data in data.items():
#     if model_name in IGNORE_MODELS:
#         continue
#     for variant in model_data.keys():
#         models.append(f"{model_name_map[model_name]}{variant_name_map[variant]}")
#         for task, task_data in model_data[variant].items():
#             tasks.add(task)
#             for metric, value in task_data.items():
#                 if isinstance(value, dict):
#                     for sub_metric in value.keys():
#                         metrics.add(f"{metric}_{sub_metric}")
#                 else:
#                     metrics.add(metric)

# # Create a dictionary to store DataFrames for each task
# task_dfs = {}

# # Fill the DataFrames with metric values for each task
# for task in sorted(tasks):
#     task_metrics = [m for m in metrics if m.split("_")[0] in task]
#     task_df = pd.DataFrame(index=sorted(task_metrics), columns=models)
    
#     for model_name, model_data in data.items():
#         if model_name in IGNORE_MODELS:
#             continue
#         for variant in model_data.keys():
#             column_name = f"{model_name_map[model_name]}{variant_name_map[variant]}"
#             if task in model_data[variant]:
#                 task_data = model_data[variant][task]
#                 for metric, value in task_data.items():
#                     if isinstance(value, dict):
#                         for sub_metric, sub_value in value.items():
#                             task_df.loc[f"{metric}_{sub_metric}", column_name] = sub_value
#                     else:
#                         task_df.loc[metric, column_name] = value
    
#     # Remove rows with all zeros
#     task_df = task_df.loc[(task_df != 0).any(axis=1)]
    
#     # Ensure that yes columns are always displayed before no columns
#     task_df = task_df[sorted(task_df.columns, key=lambda x: "General" in x)]
    
#     # Round the metric values to 4 decimal places
#     task_df = task_df.round(4)
    
#     # Create a boolean mask for rows containing "BERTScore"
#     mask = task_df.index.str.contains("BERTScore")

#     # Apply highlighting to the rows containing "BERTScore"
#     formatted_task_df = task_df.copy()
#     formatted_task_df.index = formatted_task_df.index.map(lambda x: f"\033[1m\033[92m{x}\033[0m" if "BERTScore" in x else x)
    
#     task_dfs[task] = formatted_task_df

# # Display the comparison tables for each task
# for task, task_df in task_dfs.items():
#     print(f"Comparison Table for Task: {task}")
#     print(tabulate(task_df, headers='keys', tablefmt='fancy_grid'))
#     print()

import json
import pandas as pd
from tabulate import tabulate

IGNORE_MODELS = ["llama3_exp4_1", "llama3_exp4", "llm_judge", "llama3_abl2"]
model_name_map = {
    "llama3_exp1": "Fluency ❌\n Culture ❌\n Diversity ❌",
    "llama3_exp3": "Fluency ✅\n Culture ❌\n Diversity ❌",
    "llama3_exp2": "Fluency ❌\n Culture ✅\n Diversity ❌",
    "llama3_exp4": "Fluency ❌\n Culture ❌\n Diversity ✅\n(Ultrachat)",
    "llama3_exp4_1": "Fluency ❌\n Culture ❌\n Diversity ✅\n(Ultrachat Reduced)",
    "llama3_exp4_2": "Fluency ❌\n Culture ❌\n Diversity ✅",
    "llama3_clsit": "Fluency ✅\n Culture ✅\n Diversity ✅",
    "llama3-wangchanx-demo": "WangchanX Llama3 8B",
    "llama3-typhoon-v1.5-8b": "Typhoon-v1.5 8B",
    "openthaigpt-v1-7b": "OpenThai 1.0.0 7B",
    "llama3_abl2": "Self-Instruct (Translated)",
    "llm_judge": ""
}
variant_name_map = {
    "yes": "\n(Thai Culture Test Set)",
    "no": "\n(General Test Set)"
}

# Load the JSON data from the file
with open("evaluation_results.json", "r") as file:
    data = json.load(file)

# Sort the keys in data by the order in which they appear in model_name_map
data = {k: data[k] for k in sorted(data, key=lambda x: list(model_name_map.keys()).index(x))}

# Extract the tasks, models, and metrics
tasks = set()
models = []
metrics = set()
for model_name, model_data in data.items():
    if model_name in IGNORE_MODELS:
        continue
    for variant in model_data.keys():
        models.append(f"{model_name_map[model_name]}{variant_name_map[variant]}")
        for task, task_data in model_data[variant].items():
            tasks.add(task)
            for metric, value in task_data.items():
                if isinstance(value, dict):
                    for sub_metric in value.keys():
                        metrics.add(f"{metric}_{sub_metric}")
                else:
                    metrics.add(metric)

# Create a dictionary to store DataFrames for each task
task_dfs = {}

# Fill the DataFrames with metric values for each task
for task in sorted(tasks):
    task_metrics = [m for m in metrics if m.split("_")[0] in task]
    task_df = pd.DataFrame(index=sorted(task_metrics), columns=models)
    
    for model_name, model_data in data.items():
        if model_name in IGNORE_MODELS:
            continue
        for variant in model_data.keys():
            column_name = f"{model_name_map[model_name]}{variant_name_map[variant]}"
            if task in model_data[variant]:
                task_data = model_data[variant][task]
                for metric, value in task_data.items():
                    if isinstance(value, dict):
                        for sub_metric, sub_value in value.items():
                            task_df.loc[f"{metric}_{sub_metric}", column_name] = sub_value
                    else:
                        task_df.loc[metric, column_name] = value
    
    # Remove rows with all zeros
    task_df = task_df.loc[(task_df != 0).any(axis=1)]
    
    # Ensure that yes columns are always displayed before no columns
    task_df = task_df[sorted(task_df.columns, key=lambda x: "General" in x)]
    
    # Round the metric values to 4 decimal places
    task_df = task_df.round(4)
    
    # Create a boolean mask for rows containing "BERTScore"
    mask = task_df.index.str.contains("BERTScore")
    # Apply highlighting to the rows containing "BERTScore"
    formatted_task_df = task_df.copy()
    formatted_task_df.index = formatted_task_df.index.map(lambda x: f"\033[1m\033[92m{x}\033[0m" if "BERTScore" in x else x)
    
    task_dfs[task] = formatted_task_df


# Wipe the file before writing to it
with open("task_tables.tex", "w") as file:
    file.write("")

for task, task_df in task_dfs.items():
    # Display the comparison tables for each task
    print(f"Comparison Table for Task: {task}")
    print(tabulate(task_df, headers='keys', tablefmt='fancy_grid'))
    with open("task_tables.tex", "a") as file:
        file.write(f"\\section{{{task}}}\n")
        file.write(tabulate(task_df, headers='keys', tablefmt='latex_raw'))
    print()

# Calculate the average across all tasks
average_df = pd.concat(task_dfs.values()).groupby(level=0).mean()

# Round the metric values to 4 decimal places
average_df = average_df.round(4)

# Save the average table to a LaTeX file
with open("average_table.tex", "w") as file:
    to_save = average_df.copy()

    # Remove the highlighting from the rows containing "BERTScore"
    to_save.index = to_save.index.map(lambda x: x.replace("\033[1m\033[92m", "").replace("\033[0m", ""))

    # Multiply every value by 100 (except ChrF, Squad F1)
    for i, row in average_df.iterrows():
        if "chrf" not in i.lower() and "squad" not in i.lower():
            to_save.loc[i] = row * 100
        
    # Keep two decimal places for all values (also keep zero)
    to_save = to_save.round(2)

    # Break into two tables for yes and no variants
    yes_df = to_save[[col for col in to_save.columns if "Thai Culture Test Set" in col]]
    no_df = to_save[[col for col in to_save.columns if "General Test Set" in col]]

    # Save the tables to the LaTeX file
    file.write(tabulate(yes_df, headers='keys', tablefmt='latex_raw'))
    file.write("\n\n")
    file.write(tabulate(no_df, headers='keys', tablefmt='latex_raw'))

    print("Average Table saved to average_table.tex")

# Create a boolean mask for rows containing "BERTScore"
mask = average_df.index.str.contains("BERTScore")
# Apply highlighting to the rows containing "BERTScore"
formatted_average_df = average_df.copy()
formatted_average_df.index = formatted_average_df.index.map(lambda x: f"\033[1m\033[92m{x}\033[0m" if "BERTScore" in x else x)

print("Final Average Table (Average Across All Tasks)")
print(tabulate(formatted_average_df, headers='keys', tablefmt='fancy_grid'))