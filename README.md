# We-Math Benchmarck Test

```bash
cd We-Math

# Generate single answer for each question
python3 generate/internvl2/generate.py

# Do major voting, generate 5 answers for each question
python3 generate/internvl2/generate_voting.py
```

The generated results are stored in dir `output/`

```bash
# Evaluate the 4-D metrics
python3 evalutaion/four_dimensional_metrics.py \
    --model_name InternVL2 \
    --output_json output/internvl2-base.json  \
    --main_results_csv_path result/internvl2/four_dimensional_metrics.csv

# Evaluation the accuracy
python3 evaluation/accuracy.py \
    --model_name InternVL2 \
    --output_json output/internvl2-base.json  \
    --knowledge_structure_nodes_path ../data/knowledge_structure_nodes.json \
    --main_results_csv_path result/internvl2/accuracy.csv
```
