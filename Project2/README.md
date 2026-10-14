# Singapore Rainfall Analysis & Predictive Modeling (Project 2)
Members:
- Cornelius Axel Tjandra (A0357008E)
- Atika Diana Rahardjo (A0071387Y)

This repository contains the implementation of **Task 1** (Exploratory Data Analysis & Data Profiling) and **Task 2** (30-Minute Rainfall Exceedance Prediction) using Apache Spark (PySpark).

> **Note on Implementation Formats & Group Task Distribution:**  
> Different implementation formats are used across the project due to the task distribution and collaborative workflow between group members:
> - **Task 2 (V1)** was developed as an interactive Jupyter Notebook for exploratory feature engineering and baseline model evaluation.
> - **Task 2 (V2)** was implemented by a teammate as a modular Python CLI pipeline extending upon V1's results with validation-selected regularization and model serialization.

---

## 📁 Project Structure

```text
Project2/
├── Dataset/                     # Raw CSV's (not included in submission due to size)
├── Task1/                       # Task 1: Exploratory Analysis & Profiling
│   ├── project2_part1.ipynb     # Interactive Jupyter Notebook for Task 1
│   ├── repository.py            # Complete standalone data profiling script
│   ├── task1_hourly_available_rainfall.py
│   ├── task1_monthly_rainfall.py
│   └── task1_yearly_available_rainfall.py
├── Task2/                       # Task 2: Rainfall Prediction Modeling
│   ├── V1/                      # Version 1: Baseline Notebook Implementation
│   │   └── project2_part2.ipynb # Interactive notebook (features, training, evaluation)
│   └── V2/                      # Version 2: Modular Python Pipeline (Extending V1)
│       ├── common.py            # Shared Spark session, schema, pipeline & export utilities
│       ├── 01_prepare_features.py # Stage 1: Feature computation & streaming CSV export
│       ├── 02_train_with_zero.py  # Stage 2: Train & tune LR with validation selection
│       ├── 03_evaluate_with_zero.py # Stage 3: Test evaluation (AUC, Brier score, calibration)
│       └── 04_predict.py        # Stage 4: Single-instance CLI inference tool
├── output/                      # Pre-trained model artifacts for evaluation & submission
│   └── task2/
│       ├── improved_lr_model.json
│       └── comparison/
│           └── lr_model.json    # Trained model weights & category mappings
└── README.md                    # Project documentation
```

---

## ⚙️ Prerequisites & Environment Setup

- **Python**: 3.10+ (tested with Python 3.12/3.14)
- **Java**: Java 17 or Java 21 (required by PySpark)
- **PySpark**: 3.5+ or 4.x

Install required dependencies:

```bash
pip install pyspark pandas matplotlib
```

*(Optional: install `pyarrow` for accelerated PySpark-to-Pandas conversions).*

---

## 🚀 How to Execute the Code

### Task 1: Exploratory Data Analysis & Data Profiling

Task 1 analyzes 8 years (2017–2024) of 5-minute station rainfall records across Singapore, examining completeness, spatial station coverage, time-gap anomalies, and seasonal patterns.

#### Option A: Jupyter Notebook (Recommended for visual exploration)
Open and run all cells in:
```bash
Task1/project2_part1.ipynb
```

#### Option B: Standalone Python Scripts
You can also run the modular analysis scripts directly from the project root:

```bash
# Monthly average rainfall pattern & annual seasonality plots
python Task1/task1_monthly_rainfall.py

# Yearly mean rainfall and station coverage statistics
python Task1/task1_yearly_available_rainfall.py

# Hourly rainfall availability analysis
python Task1/task1_hourly_available_rainfall.py

# Full comprehensive data profiling suite (all profiling sections)
python Task1/repository.py
```

---

### Task 2: 30-Minute Rainfall Exceedance Prediction

Task 2 predicts the probability that total rainfall in the next 30 minutes will strictly exceed a given threshold $T \in \{0.0, 0.2, 0.4, 1.0, 3.0, 24.2\}\text{ mm}$.

Two versions are provided:
- **V1 (Notebook)**: Initial interactive exploratory model development.
- **V2 (Modular Pipeline)**: Production-style Python scripts building on and refining V1 with validation-based hyperparameter tuning and model serialization.

---

#### 🔹 Task 2 — Version 1 (Jupyter Notebook)

Open and execute:
```bash
Task2/V1/project2_part2.ipynb
```
This notebook walks through:
1. Data loading and feature engineering (past 5m, 15m, 30m, 60m rainfall windows, trend, wet intervals).
2. Stratified temporal splitting: Train (2017–2021) and Test (2023–2024).
3. Candidate threshold explosion across the 6 empirical thresholds.
4. Baseline Logistic Regression training and evaluation (AUC-ROC, AUC-PR, Brier score, and persistence benchmarks).

---

#### 🔹 Task 2 — Version 2 (Modular Python Pipeline)

Run the pipeline sequentially from the **project root**:

#### Step 1: Feature Preparation
Extracts 60-minute sliding observation windows and 30-minute forward labels, streaming rows to disk without collecting into driver memory:
```bash
python Task2/V2/01_prepare_features.py
```
- **Output**: `output/task2/features.csv`
- *Note: Run once. Re-runs will protect existing data unless renamed or deleted.*

#### Step 2: Model Training & Validation Tuning
Trains logistic regression across the 6 candidate thresholds (2017–2021), tunes regularization on the 2022 validation set, and exports the fitted model:
```bash
# Default deterministic 1% sample for fast training:
python Task2/V2/02_train_with_zero.py

# Or train on a larger data fraction (e.g., 10%):
python Task2/V2/02_train_with_zero.py --fraction 0.10 --validation-fraction 0.05
```
- **Outputs**:
  - `output/task2/comparison/lr_model.json` (fitted weights and category mappings)
  - `output/task2/comparison/training_metadata.json`

#### Step 3: Model Evaluation
Evaluates test performance on unseen 2023–2024 data across each threshold, reporting log-loss, Brier score, AUC-ROC, AUC-PR, and monotonicity checks:
```bash
# Fast evaluation (Brier score, positive rate, log-loss, monotonicity):
python Task2/V2/03_evaluate_with_zero.py

# Full evaluation with ROC-AUC and PR-AUC:
python Task2/V2/03_evaluate_with_zero.py --auc

# To evaluate on a smaller sample fraction for speed:
python Task2/V2/03_evaluate_with_zero.py --fraction 0.10 --auc
```
- **Outputs**:
  - `output/task2/comparison/metrics_2023_2024.csv`
  - `output/task2/comparison/calibration_2023_2024.csv`

#### Step 4: Single-Instance Inference
Predicts the probability of exceeding any threshold for a specific weather station and historical timestamp:
```bash
python Task2/V2/04_predict.py --station S208 --time "2024-01-01 00:35:00" --threshold 3.0
```
- **Arguments**:
  - `--station`: Weather station ID (e.g. `S208`, `S50`, `S109`).
  - `--time`: Singapore local timestamp in `YYYY-MM-DD HH:MM:SS` format (5-minute aligned).
  - `--threshold`: Rainfall threshold in mm (between `0` and `25` mm).
  - `--model`: *(Optional)* Path to custom model JSON (defaults to the pre-trained `output/task2/comparison/lr_model.json`).

> *Note: The pre-trained model artifact is already included under `output/task2/comparison/lr_model.json` for submission, allowing Step 4 inference to be executed directly once features are prepared.*
