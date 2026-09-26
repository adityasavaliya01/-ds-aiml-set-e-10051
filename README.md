# Energy Demand Prediction and Clustering

## Identity

**Name:** Aditya Savaliya  
**Project:** Energy Demand Prediction and Clustering  
**Notebook:** `practical.ipynb` 

Video link- https://drive.google.com/file/d/1V38-oulzPh8zF8kg5JgOosBBFRPdsbYI/view?usp=sharing

## Objective

This project applies data science and AI/ML techniques to an energy-demand dataset. The work covers:

- Descriptive statistics and basic statistical comparison
- Data quality audit and duplicate/missing-value handling
- Train, validation, and test partitioning
- Numerical imputation and categorical encoding
- Feature engineering and standardization
- Baseline classification and Logistic Regression
- Holdout evaluation using classification metrics and a confusion matrix
- K-Means clustering and cluster selection using silhouette score
- Cluster profiling

## Dataset

The dataset is generated in the notebook and saved as:

```text
data/raw/set_e.csv
```

It contains the following main columns:

- `record_id`
- `temperature`
- `occupancy`
- `runtime`
- `load`
- `group`
- `high_demand`

The target variable is `high_demand`.

## Project Structure

```text
.
├── practical.ipynb
├── requirements.txt
├── README.md
├── data/
│   └── raw/
│       └── set_e.csv
├── splits.csv
├── logistic_predictions.csv
├── logistic_confusion_matrix.png
└── cluster_scores.csv
```

Some output files are created after running the notebook.

## Technologies and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Methodology

### 1. Data Generation

A dataset of 300 initial records is generated using NumPy. The notebook adds missing values to `temperature` and `occupancy`, creates five duplicate rows, and saves the resulting data as `data/raw/set_e.csv`.

### 2. Data Audit

The dataset contains 305 rows before duplicate removal. The audit identifies:

- 5 duplicate rows
- 15 missing values in `temperature`
- 16 missing values in `occupancy`
- 171 records with `high_demand = 1`
- 134 records with `high_demand = 0`

The duplicate rows are removed before preprocessing.

### 3. Preprocessing

The features are divided into numerical and categorical variables.

- Numerical missing values are filled using median imputation.
- `group` is encoded using One-Hot Encoding.
- An engineered feature is created:

```text
engineered_feature = load / (runtime + 1)
```

- Numerical features are standardized using `StandardScaler`.

The data is split into:

- Fit/Training: 180 records
- Validation: 60 records
- Test: 60 records

The notebook also checks that the partitions do not overlap.

### 4. Supervised Learning

A `DummyClassifier` using the most-frequent strategy is used as the baseline.

A Logistic Regression classifier is then trained and evaluated on the test set using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
