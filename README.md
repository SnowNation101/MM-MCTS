# MM-MCTS We-Math Benchmarck Test

## Preparation

### Code preparation

```bash
git clone https://github.com/SnowNation101/MM-MCTS.git
cd MM-MCTS
```

### Data preparation
Downlaod datasets

1. MathVista: https://huggingface.co/datasets/AI4Math/MathVista
2. MathVerse: https://huggingface.co/datasets/AI4Math/MathVerse
3. We-Math: https://github.com/We-Math/We-Math
4. MathVision: https://huggingface.co/datasets/MathLLMs/MathVision


Save them under `datasets/` and rename them as follow:
```bash
MM-MCTS
└── datasets
    ├── gaokao_mm
    ├── math_verse
    ├── math_vision
    ├── math_vista
    └── we_math
```
### Environment preparation

```bash
python3 -m venv mm-mcts
source mm-mcts/bin/activate
pip install -r requirements.txt
```

## Indexing
```bash
python3 util/index.py
```
The indexes may be stored under indexes dir

```

```

## Retrieve

### We-Math

```bash
python3 retrieve/we_math.py
```


## Inference

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
