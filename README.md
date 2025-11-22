# Usage Guide of `dataloader.py`

## Environment Setup

Before running the code, make sure your environment is properly set up:

- **Python 3.11 (required)**
- Use `conda` to create and activate an environment
- Install `pdftohtml` (on Mac, use Homebrew)

```bash
# Create conda environment
conda create -n env_name python=3.11
conda activate env_name

# Install required packages from requirements.txt
pip install -r requirements.txt

```



---

## 1. Load the dataset

```python
test_dataset = DocLayNetDataset(split="test")
```
> ⚠️ The first time you run it, downloading the full DocLayNet dataset may take **about 1 hour**.
---

## 2. Visualize the original labels

```python
example_idx = 0

# This will generate viz_labels.pdf showing the original labels
test_dataset.visualize_item(example_idx, output_path="viz_labels.pdf")
```

---

## 3. Visualize automatically extracted tokens (unlabeled)

```python
# All token types are set to "text" since labels are not combined yet
test_dataset.visualize_tokens(
    example_idx,
    output_path="token_viz_unlabeled.pdf",
    labels=False
)
```

Generates `token_viz_unlabeled.pdf` with the raw extracted tokens.

---

## 4. Combine dataset labels with extracted tokens

```python
features = test_dataset[example_idx]
print(features)
```

This combines the dataset labels with the tokens, giving each token the correct label.

---

## 5. Visualize tokens with correct labels

```python
# This will generate token_viz_labeled.pdf with proper labels
test_dataset.visualize_tokens(
    example_idx,
    output_path="token_viz_labeled.pdf",
    labels=True
)
```

`token_viz_labeled.pdf` effectively combines `viz_labels.pdf` and `token_viz_unlabeled.pdf`, showing tokens with the correct labels.