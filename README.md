# From-Scratch K-Nearest Neighbors Classifier

This project implements a K-nearest neighbors (KNN) classifier from scratch in Python and evaluates it on both binary and multiclass classification tasks. The project was originally developed as a course notebook and has been refactored into a repository with reusable source code, tests, and a cleaned analysis workflow.

## Project goals

- Build a KNN classifier without relying on scikit-learn's classifier implementation.
- Use Euclidean distance to classify observations by nearest-neighbor majority vote.
- Evaluate performance across different values of `k`.
- Compare binary classification performance with multiclass Iris classification.
- Discuss computational tradeoffs of brute-force KNN versus tree-based nearest-neighbor methods.

## Key results

For the binary dataset, the from-scratch KNN model achieved strong classification performance with `k=5`:

| Metric | Value |
|---|---:|
| Accuracy | 0.9800 |
| Precision | 0.98477 |
| Recall | 0.97487 |
| F1 score | 0.97980 |

On the Iris dataset, accuracy remained stable across moderate `k` values and declined sharply when `k` became too large, illustrating how high `k` values can over-smooth the decision boundary and bias predictions toward the most common training class.

## Repository structure

```text
.
├── data/
│   ├── README.md
│   └── raw/
│       ├── Iris.xlsx
│       └── Prog1data.xlsx
├── notebooks/
│   ├── Hassett_Project1_original.ipynb
│   └── knn_project_cleaned.ipynb
├── reports/
│   └── project_report.md
├── src/
│   ├── data.py
│   ├── knn.py
│   ├── metrics.py
│   ├── run_analysis.py
│   └── visualization.py
├── tests/
│   └── test_knn.py
├── requirements.txt
└── README.md
```

## Methods

The classifier stores all training observations during fitting. At prediction time, it calculates the Euclidean distance from the new observation to every training observation, identifies the `k` nearest neighbors, and assigns the majority class label.

The binary model is evaluated with accuracy, precision, recall, F1 score, and a confusion matrix. The multiclass Iris experiment evaluates training and test accuracy over a range of `k` values.

## How to run

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the analysis script:

```bash
python src/run_analysis.py
```

Or open the cleaned notebook:

```bash
jupyter notebook notebooks/knn_project_cleaned.ipynb
```

Run tests:

```bash
pytest
```

## Skills demonstrated

- Python programming
- Object-oriented model implementation
- NumPy-based distance calculations
- Data cleaning and validation
- Train/test splitting
- Binary and multiclass classification evaluation
- Matplotlib visualization
- Reproducible project organization

## Notes on Data Access

This project uses two datasets:

- **Iris dataset:** The public Iris dataset is included and used for multiclass classification.
- **Binary classification dataset:** The original binary classification dataset was provided through Purdue University’s Brightspace course site and is not included in this repository.

To respect course data-sharing restrictions, this repository includes the code and project workflow but does not redistribute the Purdue-provided dataset. Users who want to run the binary classification portion may substitute their own binary classification dataset with numeric features and class labels encoded as `+` and `-`, or adjust the preprocessing code for their chosen label format.
