# Data

This folder contains the datasets used by the project.

## Included Data

### `Iris.xlsx`

The Iris dataset is a public dataset used for multiclass classification. It is included in this repository so the Iris portion of the project can be run directly.

## Excluded Data

### `Prog1data.xlsx` / `Prog1data.csv`

The binary classification dataset used in the original course project was provided through Purdue University’s Brightspace course site. To respect course data-sharing restrictions, this dataset is **not included** in the public repository.

Students enrolled in the course should access the dataset through the official Brightspace course page.

Users outside the course may still reuse the project code by substituting their own binary classification dataset. The original workflow expects:

- numeric feature columns
- a binary target/output column
- class labels encoded as `+` and `-`

If using a different dataset or label format, update the data-loading and preprocessing steps accordingly.

## Data Storage Note

Large datasets, private datasets, course-provided datasets, and any data that cannot be publicly redistributed should not be committed to this repository.
