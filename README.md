# Hotel No-Show Prediction

## Project Description

This project implements a machine learning pipeline to predict whether a hotel booking will result in a no-show (1) or a show (0), so that the hotel chain can formulate policies that reduce the expenses incurred by empty rooms. The pipeline covers data ingestion from a CSV file, data cleaning, feature engineering, model training, hyperparameter tuning, and final evaluation on a held-out test set.

Three suitable models are evaluated and compared: Logistic Regression, K-Nearest Neighbours, and a Decision Tree Classifier. A stratified dummy classifier provides a naive baseline.

---

## Prerequisites and Installation Instructions

### Requirements

* Anaconda or Miniconda
* Python 3.11
* Required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-folder>
```

2. Create and activate a conda environment:
```bash
conda create -n aiapenv python=3.11
conda activate aiapenv
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

---

## Instructions for Executing the Pipeline

To run the complete pipeline, execute:

```bash
bash run.sh
```

Or directly:

```bash
python main.py
```

The exploratory analysis and all supporting plots are in `eda.ipynb`. To execute the notebook in place:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=aiap eda.ipynb
```

---

## Description of Logical Steps/Flow of the Pipeline

### Logical Flow of the Pipeline

The main script orchestrates the entire pipeline by:

* Loading configuration and reading the booking CSV
* Running data preparation, cleaning and feature engineering
* Splitting the cleaned data into training, validation, and test sets with stratification
* Training and evaluating baseline models
* Tuning all three model families with grid search
* Selecting and evaluating the best model on the held-out test set

### 1. Configuration (`config.yaml`)

The pipeline loads configuration from `src/config.yaml`, which specifies:

* CSV file path
* Target column name (`no_show`)
* Feature categorisation (numerical and nominal)
* Train/validation/test split ratios
* Hyperparameter grids for model tuning
* Cross-validation folds and scoring metric (`f1` for the positive no-show class)

### 2. Data Ingestion (`main.py`)

The raw data is a single CSV of 119,391 hotel bookings with 15 columns. It is read with `pandas.read_csv` and passed to `DataPreparation`. This is the only data source; no external tables or joins are required.

### 3. Data Preparation (`DataPreparation` class)

The `DataPreparation` class handles all cleaning and feature engineering:

**Cleaning:**

* Drop the single row where `no_show` is null, as the target cannot be imputed
* Drop `booking_id`, which is a unique row identifier with no predictive value
* Standardise `booking_month`, `arrival_month` and `checkout_month`: strip whitespace, lowercase, map through a month number and back to a Title-case month name. This repairs the 89 distinct `arrival_month` spellings caused by mixed casing
* Map the word-form adult counts in `num_adults` (`'one'` to 1, `'two'` to 2) and coerce the column to numeric
* Parse `price` from strings such as `SGD$ 492.98` and `USD$ 1351.22` into a float, then convert USD amounts to SGD at an assumed rate of **1 USD = 1.35 SGD**. The result is stored as `price_sgd`. The exchange rate is an explicit assumption: the data records two currencies without a conversion date, and 1.35 is a reasonable planning rate
* Fill missing `room` values with an explicit `'Unknown'` category. Missingness is not predictive (36.8% no-show when room is missing versus 37.1% when present), but 21,613 rows is too many to drop and inventing a room type would be misleading
* Impute `price_sgd` with the median price of the same `(branch, room)` group, falling back to the overall median. Missingness is again not predictive (37.2% versus 37.0%), and price depends strongly on branch and room, so a group median is a better estimate than dropping 24,882 rows
* Drop `checkout_month`, `checkout_day`, `price` and `booking_month` once their information is captured by the engineered features

**Feature Engineering:**

* `stay_length` — `checkout_day - arrival_day`, with the number of days in the arrival month added when the difference is negative. Negative differences occur because cross-month checkouts were recorded as negative day offsets: a day-30 arrival with a day-1 checkout in the next month appears as `-29`, and adding 31 days recovers the true 2-night stay. The result is floored at one night, since a hotel stay cannot have zero or negative duration
* `booking_month_num` and `arrival_month_num` — the month positions as 1-12 ordinals
* `booking_lead_months` — `(arrival_month_num - booking_month_num) % 12`, so a November booking for a January arrival is 2 months ahead rather than -10

**Column selection:** the DataFrame is filtered to retain only the columns declared in `config.yaml` plus the target, dropping everything else before training.

### 4. Model Training (`ModelTraining` class)

The `ModelTraining` class manages:

* Splitting cleaned data into training (80%), validation (10%) and test (10%) sets with stratification on the target
* Training three baseline classifiers: Logistic Regression, K-Nearest Neighbours, and a Decision Tree
* Hyperparameter tuning via `GridSearchCV` (5-fold CV, scoring on F1 for the positive class) across the same three model families
* Evaluating all models on the validation set using Accuracy, Precision, Recall, F1 Score, and ROC AUC for the no-show class
* Selecting the best overall model by validation F1 and evaluating it once on the held-out test set

### 5. Preprocessing Pipeline (`DataPreparation._create_preprocessor`)

A `ColumnTransformer` is constructed with:

* **Numerical** (`StandardScaler`): `price_sgd`, `num_adults`, `num_children`, `arrival_day`, `stay_length`, `booking_lead_months`, `booking_month_num`, `arrival_month_num`
* **Nominal** (`OneHotEncoder`, `drop='first'`, `handle_unknown='ignore'`): `branch`, `country`, `first_time`, `room`, `platform`
* `remainder='drop'` — any column not explicitly listed is excluded from the model

### 6. Why Resampling Was Not Applied

The target split is 63.0% show / 37.0% no-show. This is a moderate imbalance, not a severe one: the minority class contains 44,224 training examples, which is enough for all three models to learn the positive pattern without synthetic data.

Resampling was also deliberately avoided because it would distort the predicted probabilities. The no-show policy is threshold-based, and a model trained on a 50/50 resampled set would output probabilities that no longer reflect the real booking population. Keeping the natural prior means a predicted probability of 0.6 can be read as an actual 60% chance of a no-show. Stratified splitting preserves the 63/37 mix in the training, validation and test sets, and F1 is used for model comparison so that the majority class does not dominate the selection.

---

## Overview of Key Findings from EDA

### Target Variable Characteristics

The target `no_show` is binary and moderately imbalanced:

| Outcome | Count | Share |
|---------|-------|-------|
| Show (0) | 75,166 | 63.0% |
| No-Show (1) | 44,224 | 37.0% |

Because the majority class is 63%, a constant "show" predictor would already reach 63% accuracy. Accuracy is therefore reported alongside precision, recall, F1 and ROC AUC, and F1 is the primary selection metric.

### Key Predictors of No-Show

**Strong signals (clear trends in bivariate analysis):**
- **Country** — the strongest categorical signal. Chinese guests no-show 56.6% of the time, while Japanese guests are at 17.1%. Indonesia, India and Australia sit in between
- **Branch** — Changi runs at 41.7% no-shows against Orchard's 27.8%
- **First-time guest status** — first-time guests no-show 37.8% of the time versus 14.5% for returning guests
- **Room type** — King rooms are the highest at 39.2%, Queen the lowest at 31.4%
- **Booking lead time** — same-month bookings no-show 25.2% of the time, rising steadily to 41.2% for bookings made 9-11 months ahead
- **Price** — the 801-1000 SGD band peaks at 42.6%, while the 1200+ band is lowest at 27.5%
- **Stay length** — one-night stays are lowest at 32.7%, two- and three-night stays rise to about 41%

**Weak signals:**
- **Platform** — essentially flat, with all four channels between 36% and 37%
- **Booking month** — June is highest at 40.9% and October lowest at 36.2%, a real but modest seasonal effect
- **Number of adults and children** — near-identical no-show rates across categories

### Missingness and Duplicates

`booking_id` is unique with no duplicates. Apart from one fully-null row, only `room` (21,613 nulls) and `price` (24,882 nulls) have substantial missingness. Neither missingness pattern is informative about the target, which is why both were imputed rather than used as a reason to drop roughly 20% of the data.

### Correlation Analysis

Spearman correlation was used over Pearson because the target is binary and the numeric features are skewed. Pearson assumes continuous, normally distributed variables and measures linear association, which is not appropriate for a 0/1 outcome. The strongest numeric correlations with `no_show` are modest: `booking_lead_months` at about +0.09, `stay_length` at about +0.06 and `price_sgd` at about -0.05. The predictive signal therefore lives mainly in the categorical splits, which is why a multivariable model is necessary.

---

## Feature Handling Description

### Data Cleaning

* The target null row was dropped because the target cannot be imputed
* `booking_id` was dropped as an identifier
* Month names were standardised through a month-number mapping, removing 89 casing variants from `arrival_month`
* Word-form adult counts were mapped to integers and the column coerced to numeric
* Prices were parsed from their currency-prefixed strings and converted to a single SGD scale at 1 USD = 1.35 SGD
* Missing room values were filled with an explicit `'Unknown'` category
* Missing prices were imputed with the median of the same `(branch, room)` group, falling back to the overall median
* Cross-month stay lengths were repaired by adding the number of days in the arrival month, then floored at one night

### Feature Processing

* **Numerical features** — standardised with `StandardScaler` because price is on a scale of hundreds to thousands of SGD while the counts are single digits. Scaling is essential for KNN's distance calculations and keeps logistic regression coefficients comparable
* **Nominal features** — one-hot encoded with `drop='first'` to avoid perfect multicollinearity, and `handle_unknown='ignore'` so unseen categories at inference time do not raise errors
* **Binned columns** — created only for EDA plots and excluded from the model by the explicit feature lists and `remainder='drop'`

---

## Explanation of Model Choices

### Models Implemented and Justifications

1. **Logistic Regression** — a linear model with a probabilistic output. It is the natural benchmark for a binary outcome, is well calibrated, and its coefficients show the direction of each feature. If a simple linear boundary performs nearly as well as a non-linear model, the extra complexity of the latter is harder to justify.
2. **K-Nearest Neighbours** — a distance-based, non-parametric classifier. It makes no assumption about the shape of the decision boundary and can capture local clusters of similar bookings. It is sensitive to feature scaling, which the preprocessing pipeline handles, and it provides a useful contrast to the linear model.
3. **Decision Tree Classifier** — a non-linear model that combines several features through if-then splits. It captures interactions such as country combined with branch, and its structure can be plotted and read as business rules. Pre-pruning through `max_depth` and `min_samples_leaf` controls overfitting.

A **stratified dummy classifier** is included as the naive baseline, representing random guessing that respects the 63/37 prior.

### Hyperparameter Tuning

* Grid search is performed for all three models with 5-fold cross-validation
* Scoring is F1 for the positive (no-show) class, so precision and recall are balanced rather than the majority class dominating
* The logistic regression grid varies regularisation strength `C`; the KNN grid varies the number of neighbours, weighting and distance metric; the decision tree grid varies the split criterion, depth and minimum leaf size
* Models are fitted on the natural, unresampled training set so the tuned probabilities remain representative of the booking population

---

## Evaluation of Models

### Evaluation Metrics

All metrics are computed for the positive class (`no_show = 1`). ROC AUC uses the predicted probability of a no-show.

* **Accuracy** — proportion of bookings correctly classified
* **Precision** — of the bookings flagged as no-shows, the share that actually no-show
* **Recall** — of the bookings that no-show, the share that were flagged
* **F1 Score** — harmonic mean of precision and recall; the primary selection metric
* **ROC AUC** — class separation ability across all thresholds, independent of the 0.5 cut-off

### Model Performance Comparison

#### Baseline Models (Before Hyperparameter Tuning)

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Dummy Classifier (stratified) | 0.54058 | 0.37737 | 0.36943 | 0.37336 | 0.50537 |
| Logistic Regression | 0.72527 | 0.66503 | 0.52069 | 0.58407 | 0.74876 |
| K-Nearest Neighbours | 0.70148 | 0.61329 | 0.52566 | 0.56611 | 0.71497 |
| Decision Tree | 0.73524 | 0.69415 | 0.51006 | 0.58804 | 0.75827 |

The dummy classifier is close to the 0.534 accuracy expected from drawing randomly from the class prior, and its ROC AUC is essentially 0.50. All three real models clear it comfortably, which confirms that the booking attributes contain learnable signal. Logistic Regression and the shallow Decision Tree are close at about 0.73 accuracy, with the tree slightly ahead on precision and ROC AUC. KNN is the weakest baseline at 0.70 accuracy.

#### Hyperparameter Tuned Models (After Grid Search)

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Tuned Logistic Regression | 0.72527 | 0.66503 | 0.52069 | 0.58407 | 0.74876 |
| Tuned KNN | 0.73365 | 0.67346 | 0.54556 | 0.60280 | 0.76272 |
| Tuned Decision Tree | 0.73515 | 0.66711 | 0.56907 | 0.61420 | 0.77087 |

Best grid search parameters:

| Model | Selected Hyperparameters | Best CV F1 |
| :--- | :--- | :---: |
| Logistic Regression | `C=10`, `solver='liblinear'`, `max_iter=1000` | 0.57301 |
| KNN | `n_neighbors=15`, `weights='distance'`, `metric='manhattan'` | 0.59714 |
| Decision Tree | `criterion='gini'`, `max_depth=None`, `min_samples_leaf=10` | 0.61582 |

Tuning improves KNN markedly by moving away from a very local five-neighbour model, and improves the tree by keeping an unbounded depth while requiring at least 10 bookings per leaf. Logistic Regression does not improve because its best regularisation strength is at the edge of the grid; the linear boundary is already close to its ceiling.

#### Final Model on the Held-Out Test Set

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Tuned Decision Tree | 0.74085 | 0.67301 | 0.58412 | 0.62542 | 0.77728 |

The test scores are slightly higher than the validation scores but follow the same pattern, which suggests the model is not overfitted to the validation split.

---

## Final Model Selection and Interpretation

The **Tuned Decision Tree** was selected as the most suitable model. It achieved the highest validation F1 (0.61420), the highest validation ROC AUC (0.77087) and the highest validation accuracy (0.73515), and it remained the strongest model on the held-out test set with an F1 of 0.62542 and a ROC AUC of 0.77728.

Key observations from the evaluation:

1. The margin over tuned KNN is about 0.011 F1 and 0.008 ROC AUC. The tree is ahead but not decisively so, so the choice is also supported by operational considerations: the tree is a set of readable if-then rules, stores only split thresholds rather than the training set, and produces risk bands that follow the same country, branch and first-time factors the business already understands.
2. All three models agree on the broad ranking of features. Country, branch and first-time status dominate the tree's first splits, which matches the bivariate analysis and makes the model easier to trust.
3. The performance ceiling is about 0.62 F1. No-show behaviour has an irreducible random component, and the available features describe the booking rather than the guest's changing circumstances after booking. Further gains are more likely to come from richer behavioural data than from further tuning of the same three algorithms.

### Business Interpretation for a No-Show Policy

The two error types have asymmetric costs:

* **False negative** — a guest no-shows without being flagged. The room stays empty and the hotel loses the revenue it could have recovered through overbooking or a reminder campaign.
* **False positive** — a guest is flagged but arrives. If the hotel acts aggressively, it may deny a room to a genuine guest, causing a service failure and reputational damage.

The model's default 0.5 threshold leans towards precision (0.673) over recall (0.584), which is the conservative choice: it flags only bookings it is reasonably confident about, so the hotel can target reminders and selective overbooking without routinely punishing reliable guests.

A practical policy would use probability tiers rather than a single cut-off:

* **High risk (above roughly 0.6):** send a confirmation or deposit request, and consider limited overbooking against these rooms
* **Medium risk (roughly 0.4 to 0.6):** send a reminder and set a cancellation deadline
* **Low risk (below 0.4):** no intervention

This lets the hotel match the intervention to the cost of being wrong. It also acknowledges the model's limits: with F1 around 0.63, a substantial share of no-shows will still be missed and some flagged guests will still arrive. The model is a decision-support tool for allocating attention and overbooking risk, not an automatic denial system.

---

## Considerations for Deployment

### Model Persistence

To make this project production-ready, implement model persistence:

1. Add model saving functionality to the `ModelTraining` class:
```python
import joblib

def save_model(self, model, filename):
    """Save the trained model to a file."""
    joblib.dump(model, filename)

def load_model(self, filename):
    """Load a trained model from a file."""
    return joblib.load(filename)
```

2. Create a separate inference script (`inference.py`) for making predictions with saved models:
```python
import pandas as pd
import joblib
from src.data_preparation import DataPreparation

# Load configuration and model
# Preprocess new data
# Make predictions
```

### Operational Considerations

* **Retraining cadence** — booking behaviour and market mix change over time. The model should be retrained on a rolling window, and the validation F1 and ROC AUC should be monitored for drift
* **Threshold maintenance** — the optimal probability threshold depends on room rates, occupancy and the cost of denied arrivals. It should be reviewed by revenue management rather than fixed once
* **Unknown categories** — the encoder uses `handle_unknown='ignore'`, so new countries or platforms will not break inference, but a sudden influx of an unseen category should trigger a review of the encoding
* **Fairness and explainability** — country and first-time status are strong predictors, and a policy that denies rooms based on them could be perceived as unfair. The model should inform reminders and deposits rather than blanket denials, and the business should be able to explain every intervention to the guest
* **Data quality** — the negative `checkout_day` values show that the upstream booking system can emit inconsistent date fields. The cleaning logic repairs them, but the source system should also be fixed so future extracts do not require the same correction
