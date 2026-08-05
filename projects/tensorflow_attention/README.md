# TensorFlow temporal attention

## Goal

Compare a convolutional baseline with cross-attention between absolute signal levels and temporal differences for SMAD condition classification.

## Contents

- `notebooks/attention_model.ipynb`: baseline, attention experiment and comparison
- `src/tensorflow_attention/model.py`: both model builders and an attention representation probe
- `src/tensorflow_attention/training.py`: function-based Keras training with early stopping

## Run

```bash
python -m pip install -e ".[tensorflow]"
jupyter lab projects/tensorflow_attention/notebooks/attention_model.ipynb
```

The post-attention representation is a model diagnostic, not a causal explanation of the underlying biology.
