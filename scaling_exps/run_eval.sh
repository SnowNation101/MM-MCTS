#!/bin/bash

model_name="internvl2"

for scaling in 1 2 4 8 16 32; do
    output_json="${model_name}/majority_voting/${model_name}-${scaling}.json"

    # accuracy.py
    python3 ../evaluation/accuracy.py \
        --model_name="$model_name" \
        --output_json="$output_json" \
        --knowledge_structure_nodes_path="../datasets/we_math/knowledge_structure_nodes.json" \
        --main_results_csv_path="${model_name}/majority_voting/accuracy-${scaling}.csv"

    # four_dimensional_metrics.py
    python3 ../evaluation/four_dimensional_metrics.py \
        --model_name="$model_name" \
        --output_json="$output_json" \
        --main_results_csv_path="${model_name}/majority_voting/four_dimensional_metrics-${scaling}.csv"
done

for scaling in 2 4 8 16 32; do
    output_json="${model_name}/orm/${model_name}-${scaling}.json"

    # accuracy.py
    python3 ../evaluation/accuracy.py \
        --model_name="$model_name" \
        --output_json="$output_json" \
        --knowledge_structure_nodes_path="../datasets/we_math/knowledge_structure_nodes.json" \
        --main_results_csv_path="${model_name}/orm/accuracy-${scaling}.csv"

    # four_dimensional_metrics.py
    python3 ../evaluation/four_dimensional_metrics.py \
        --model_name="$model_name" \
        --output_json="$output_json" \
        --main_results_csv_path="${model_name}/orm/four_dimensional_metrics-${scaling}.csv"
done

for scaling in 2 4 8 16 32; do
    output_json="${model_name}/random/${model_name}-${scaling}.json"

    # accuracy.py
    python3 ../evaluation/accuracy.py \
        --model_name="$model_name" \
        --output_json="$output_json" \
        --knowledge_structure_nodes_path="../datasets/we_math/knowledge_structure_nodes.json" \
        --main_results_csv_path="${model_name}/random/accuracy-${scaling}.csv"

    # four_dimensional_metrics.py
    python3 ../evaluation/four_dimensional_metrics.py \
        --model_name="$model_name" \
        --output_json="$output_json" \
        --main_results_csv_path="${model_name}/random/four_dimensional_metrics-${scaling}.csv"
done