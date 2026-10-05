# Construction Permit Delay Intelligence

An end-to-end data engineering, exploratory analytics, machine learning, and visualization project analyzing construction permit processing delays in Washington, DC.

The project asks a practical question:

> **Can we identify patterns associated with construction-permit processing delays and predict which completed permits are likely to exceed a defined processing-time threshold?**

Using public construction permit data from the District of Columbia, the project builds a complete analytics workflow in **Databricks and PySpark**, stores data using **Delta Lake**, develops a baseline predictive model using **Spark MLlib**, and communicates the results through **Tableau**.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Key Results](#key-results)
* [Project Architecture](#project-architecture)
* [Dataset](#dataset)
* [Data Engineering Pipeline](#data-engineering-pipeline)

  * [Bronze Layer](#bronze-layer)
  * [Silver Layer](#silver-layer)
  * [Gold Layer](#gold-layer)
* [Data Quality and Missingness](#data-quality-and-missingness)
* [Defining Permit Delays](#defining-permit-delays)
* [Exploratory Data Analysis](#exploratory-data-analysis)
* [Temporal Analysis and Observation-Window Effects](#temporal-analysis-and-observation-window-effects)
* [Machine Learning](#machine-learning)

  * [Modeling Objective](#modeling-objective)
  * [Modeling Cohort](#modeling-cohort)
  * [Feature Engineering](#feature-engineering)
  * [Train/Test Split](#traintest-split)
  * [Model Pipeline](#model-pipeline)
  * [Model Performance](#model-performance)
  * [Confusion Matrix](#confusion-matrix)
* [Model Interpretation and Limitations](#model-interpretation-and-limitations)
* [Tableau Dashboard](#tableau-dashboard)
* [Key Findings](#key-findings)
* [Technology Stack](#technology-stack)
* [Repository Structure](#repository-structure)
* [Reproducibility](#reproducibility)
* [Future Improvements](#future-improvements)
* [Project Status](#project-status)

---

# Project Overview

Construction permit processing can vary substantially from one application to another.

Some permits are processed within days, while others remain in the system for months or even years.

This project investigates whether publicly available permit information can be used to:

1. Measure construction permit processing times.
2. Identify patterns associated with unusually long processing times.
3. Define a practical binary delay classification.
4. Build a predictive baseline for identifying permits likely to exceed that threshold.
5. Communicate the findings through an interactive business-intelligence dashboard.

The project intentionally combines **data engineering, analytics, machine learning, and visualization** rather than treating machine learning as an isolated modeling exercise.

The complete workflow is:

```text
Public DC Construction Permit Data
                |
                v
           Databricks
                |
                v
        Bronze Delta Layer
                |
                v
        Silver Delta Layer
                |
                v
         Gold Delta Layer
                |
                v
      Exploratory Data Analysis
                |
                v
       Feature Engineering
                |
                v
      Spark MLlib Pipeline
                |
                v
      Logistic Regression
                |
                v
       Tableau Visualization
```

---

# Key Results

The completed-permit dataset contains:

| Metric                  |         Result |
| ----------------------- | -------------: |
| Original records        |     **27,442** |
| Completed permits       |     **17,365** |
| Median processing time  |     **9 days** |
| Mean processing time    |  **42.3 days** |
| Maximum processing time | **1,174 days** |
| Delay threshold         |   **>40 days** |
| Overall delay rate      |      **24.9%** |

The baseline Logistic Regression model achieved the following results on a held-out 2025 test set:

| Metric                  |    Result |
| ----------------------- | --------: |
| ROC-AUC                 | **0.831** |
| F1                      | **0.796** |
| Weighted Recall         | **0.801** |
| Delayed-class Precision | **83.9%** |
| Delayed-class Recall    | **89.0%** |

These results indicate that the available permit, applicant, geographic, and temporal features contain meaningful predictive signal for identifying permits likely to exceed the 40-day threshold.

---

# Project Architecture

The project follows a **Bronze / Silver / Gold** data architecture.

```text
                       DC Open Data
                            |
                            v
                    +---------------+
                    |    Databricks |
                    +---------------+
                            |
                            v
                    +---------------+
                    |    Bronze     |
                    |  Raw / Delta  |
                    +---------------+
                            |
                            v
                    +---------------+
                    |    Silver     |
                    | Cleaned Data  |
                    +---------------+
                            |
                            v
                    +---------------+
                    |     Gold      |
                    | Model-Ready   |
                    +---------------+
                            |
                  +---------+---------+
                  |                   |
                  v                   v
           Exploratory Analysis   ML Modeling
                  |                   |
                  +---------+---------+
                            |
                            v
                    Tableau Dashboard
```

This architecture separates raw ingestion from data cleaning and analytical modeling.

It also makes the workflow easier to reproduce and extend.

---

# Dataset

The project uses the **District of Columbia Open Data construction permit dataset**.

The source data contains construction permit records including dates, permit information, applicant information, geographic information, and permit activity indicators.

The original dataset contains:

**27,442 records**

The project does not include the raw dataset in the GitHub repository.

This keeps the repository lightweight and avoids committing a large source-data file unnecessarily.

---

# Data Engineering Pipeline

## Bronze Layer

The Bronze layer contains the initial raw dataset stored in Databricks as a Delta table.

Table:

```text
bronze_construction_permits
```

The Bronze layer intentionally applies minimal transformation.

Its purpose is to provide a persistent representation of the source dataset before analytical transformations are applied.

The raw data is written using Delta Lake:

```python
df_raw.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("bronze_construction_permits")
```

---

# Silver Layer

The Silver layer standardizes and cleans the Bronze data.

Major transformations include:

* Normalizing column names
* Converting date fields to proper date types
* Calculating permit processing time
* Inspecting missing values
* Inspecting categorical distributions
* Performing data-quality checks
* Standardizing fields for downstream analysis

Column names were normalized to lowercase and standardized to make the dataset easier to work with in PySpark.

Processing time was calculated as:

```text
processing_days = issue_date - application_date
```

using Spark's `datediff` function.

The Silver table contains all **27,442 original records**, including records where processing time cannot be calculated.

Table:

```text
silver_construction_permits
```

---

# Gold Layer

The Gold layer represents the analytical dataset used for exploratory analysis and machine learning.

Only records with both an application date and an issue date are retained for processing-time analysis.

This produces:

**17,365 completed permits**

The Gold layer also introduces derived analytical features.

### Temporal Features

* `application_year`
* `application_month`
* `application_day`
* `application_day_of_week`
* `application_quarter`

### Geographic Features

* `latitude`
* `longitude`

### Permit Activity Features

* `isexcavation`
* `isfixture`
* `ispaving`
* `islandscaping`
* `isprojections`

### Applicant and Permit Information

* `permitteename`
* `ownername`
* `applicantcompanyname`

### Target Variable

* `delayed`

The resulting Gold table is:

```text
gold_construction_permits
```

---

# Data Quality and Missingness

The original dataset contains records with incomplete date information.

The date-quality breakdown is:

| Record Group                     |      Count |
| -------------------------------- | ---------: |
| Application + issue date present | **17,365** |
| Application date only            |  **3,524** |
| Neither date present             |  **6,553** |
| Total                            | **27,442** |

Records without an issue date cannot provide an observed completed processing duration.

Therefore, the primary processing-time analysis is restricted to permits with both required dates.

This distinction is important because excluding incomplete records is not equivalent to saying those permits were processed quickly or slowly. Their final processing times are simply not yet observable in the dataset.

---

# Defining Permit Delays

The project defines a permit as delayed when its observed processing time exceeds **40 days**.

```text
delayed = 1 if processing_days > 40
delayed = 0 otherwise
```

The threshold was selected using the empirical distribution of completed permits.

The 75th percentile of processing time is approximately:

**40 days**

Therefore, the threshold identifies approximately the slowest quarter of completed permits.

This creates a practical binary classification problem while preserving the original continuous processing-time measure for exploratory analysis.

---

# Exploratory Data Analysis

The completed-permit population contains:

| Metric             |      Value |
| ------------------ | ---------: |
| Count              |     17,365 |
| Mean               | 42.25 days |
| Standard deviation | 96.96 days |
| Minimum            |     0 days |
| 25th percentile    |     2 days |
| Median             |     9 days |
| 75th percentile    |    40 days |
| Maximum            | 1,174 days |

The distribution is strongly right-skewed.

The median processing time is only **9 days**, while the mean is approximately **42 days**.

The difference occurs because a smaller group of permits takes substantially longer to process.

The long tail extends to more than three years.

This makes the median more representative of the typical permit than the mean.

---

# Delay Distribution

The completed permits divide into:

| Classification |      Count |
| -------------- | ---------: |
| Not delayed    | **13,044** |
| Delayed        |  **4,321** |

The resulting delay rate is:

**24.9%**

This is consistent with the threshold being approximately the 75th percentile of processing time.

---

# Temporal Analysis and Observation-Window Effects

One of the most important findings from the analysis is that permit processing times cannot be interpreted purely as a simple year-over-year trend.

The observed completed-permit population is affected by the amount of time available for permits to finish.

For example, a permit submitted recently cannot yet appear as a completed permit with a processing duration of several hundred days.

This produces an **observation-window effect**, sometimes described as right-censoring or cohort selection.

## Why this matters

The dataset contains substantially more completed permits from 2026 than from earlier years.

However, recent 2026 applications have had less time to complete.

Consequently, the completed 2026 population is disproportionately composed of permits that have already finished relatively quickly.

The analysis therefore found that the observed decline in processing time in recent cohorts should **not** automatically be interpreted as evidence that permit processing became dramatically faster.

The observed data is conditional on completion.

This is an important analytical limitation because otherwise a model or dashboard could incorrectly present recent permit cohorts as inherently faster.

---

# Modeling Cohort Selection

The analysis found major differences between application-year cohorts.

Earlier years contained relatively few observations:

| Year | Completed Permits |
| ---- | ----------------: |
| 2023 |                18 |
| 2024 |               319 |
| 2025 |             1,598 |
| 2026 |            15,430 |

Because the earlier cohorts were small and the 2026 cohort is strongly affected by the observation window, **2025 was selected as the primary modeling cohort**.

This provides a more substantial and comparatively mature population while avoiding the most severe recent-cohort effects.

---

# Machine Learning

## Modeling Objective

The machine-learning objective is:

> Predict whether a completed construction permit will exceed the 40-day processing-time threshold.

This is formulated as a binary classification problem.

```text
0 = Not delayed
1 = Delayed
```

The project uses **Logistic Regression** as the baseline model.

The objective was not to build the most sophisticated possible model.

Instead, the goal was to establish a transparent, reproducible baseline using Spark MLlib.

---

# Modeling Cohort

The primary modeling population consists of **2025 permits**.

The 2025 cohort contains:

**1,598 completed permits**

The dataset was split into training and test populations using an 80/20 random split with a fixed seed.

The resulting datasets were:

| Dataset  |   Records |
| -------- | --------: |
| Training | **1,292** |
| Test     |   **306** |

---

# Feature Engineering

The model uses a combination of temporal, geographic, permit-characteristic, and applicant-related features.

## Numeric Features

```text
application_month
application_day
application_day_of_week
application_quarter
latitude
longitude
```

## Categorical Features

```text
isexcavation
isfixture
ispaving
islandscaping
isprojections
permitteename
ownername
applicantcompanyname
```

The `ispsrental` field was excluded because it contained extremely few positive observations.

The contractor field was also excluded because it contained substantial missingness.

Identifier fields and fields that could introduce leakage were excluded from the modeling feature set.

---

# Geographic Data Cleaning

Latitude and longitude were originally represented as string fields.

They were converted to numeric values before modeling.

Invalid geographic values were removed by restricting the coordinates to a reasonable geographic range around Washington, DC.

Missing valid coordinates were handled using median imputation inside the Spark ML pipeline.

This prevents missing geographic values from causing the model pipeline to fail while avoiding manual replacement of the original data.

---

# Categorical Encoding

Categorical variables were processed using Spark MLlib.

The pipeline uses:

1. `StringIndexer`
2. `OneHotEncoder`
3. Feature assembly
4. Logistic Regression

The indexers use:

```text
handleInvalid = keep
```

This allows the model to handle previously unseen or invalid categorical values during transformation rather than failing.

---

# Model Pipeline

The Spark ML pipeline follows this general structure:

```text
Raw Features
     |
     v
StringIndexer
     |
     v
OneHotEncoder
     |
     v
Median Imputation
     |
     v
VectorAssembler
     |
     v
Logistic Regression
     |
     v
Prediction
```

The model was trained using Spark MLlib's Logistic Regression implementation.

The model was configured with a maximum of 100 iterations.

---

# Model Performance

The model achieved the following performance on the held-out 2025 test set:

| Metric          |     Score |
| --------------- | --------: |
| ROC-AUC         | **0.831** |
| F1              | **0.796** |
| Weighted Recall | **0.801** |

The ROC-AUC of **0.831** indicates useful separation between delayed and non-delayed permits.

The F1 score of **0.796** indicates a reasonably strong balance between precision and recall for the binary classification task.

---

# Delayed-Class Performance

Because identifying delayed permits is the primary operational objective, the delayed class was examined separately.

| Metric    |     Score |
| --------- | --------: |
| Precision | **83.9%** |
| Recall    | **89.0%** |

A delayed-class recall of approximately 89% means the model identified most of the delayed permits in the held-out test set.

The tradeoff is that some permits predicted as delayed were actually below the 40-day threshold.

---

# Confusion Matrix

The held-out test results produced the following confusion matrix:

```text
                         Predicted
                    Not Delayed   Delayed

Actual Not Delayed       52          37
Actual Delayed           24         193
```

This gives:

* True Negatives: **52**
* False Positives: **37**
* False Negatives: **24**
* True Positives: **193**

The model therefore captured the majority of delayed permits while producing a moderate number of false positives.

---

# Model Interpretation and Limitations

The model demonstrates predictive association, not causation.

For example, if a particular applicant or permit characteristic is associated with higher predicted delay probability, that does not establish that the characteristic causes the delay.

The model should therefore be interpreted as a **risk-screening tool**, not a causal explanation of the permitting process.

Several limitations are important.

## 1. Observation-window bias

Recent permits are less likely to have accumulated long processing durations because they have had less time to complete.

## 2. Completed-permit selection

The modeling dataset only contains permits for which an issue date is observed.

Permits still being processed are excluded because their final processing time is unknown.

## 3. Limited historical data

Earlier cohorts contain relatively few observations compared with 2025 and 2026.

## 4. Baseline model

Logistic Regression was intentionally used as a baseline.

More sophisticated models could potentially improve predictive performance.

## 5. No causal inference

The model identifies statistical relationships rather than proving why delays occur.

## 6. Threshold dependency

The classification target depends on the selected 40-day threshold.

A different operational definition of delay would produce a different target and potentially different model performance.

---

# Tableau Dashboard

The project includes a Tableau dashboard designed to communicate the most important findings to a non-technical audience.

The dashboard focuses on operationally meaningful questions rather than model internals.

Planned dashboard components include:

### Processing Time Distribution

Shows the distribution of completed permit processing times and highlights the 40-day delay threshold.

### Delay Rate Over Time

Shows how the observed delay rate changes across application periods.

### Delay Rate by Year

Compares completed-permit delay rates across application cohorts.

### Delay Rate by Permit Activity

Examines delay rates across permit activity characteristics such as excavation, paving, fixture, landscaping, and projection indicators.

### Key Performance Indicators

The dashboard includes:

* **17,365** completed permits
* **24.9%** overall delay rate
* **9 days** median processing time
* **0.831** Logistic Regression ROC-AUC

The ROC-AUC KPI refers specifically to the held-out 2025 model evaluation and should not be interpreted as an overall population statistic.

---

# Key Findings

## Finding 1: The typical completed permit is processed quickly

The median processing time is only **9 days**.

This means that half of the completed permits in the analytical population were processed within nine days.

However, the mean is much higher at approximately **42 days**, demonstrating the impact of the long right tail.

---

## Finding 2: A substantial minority experience long processing times

Approximately **24.9%** of completed permits exceeded the 40-day threshold.

That corresponds to:

**4,321 delayed permits**

out of:

**17,365 completed permits**.

---

## Finding 3: Processing times are highly variable

The maximum observed processing time was:

**1,174 days**

This is more than three years.

The difference between the median and maximum demonstrates that a small group of permits experiences dramatically longer processing durations.

---

## Finding 4: Recent-year comparisons are misleading without accounting for the observation window

The data initially appears to show a dramatic improvement in processing times in 2026.

However, recent applications have had less time to become long-duration completed permits.

Therefore, completed-permit statistics from recent cohorts are not directly comparable with mature historical cohorts.

This is one of the most important analytical conclusions of the project.

---

## Finding 5: Applicant, permit, geographic, and temporal information contains predictive signal

The Logistic Regression model achieved:

**0.831 ROC-AUC**

on held-out 2025 data.

This indicates that the available features provide meaningful information for distinguishing permits likely to exceed the 40-day threshold.

---

# Technology Stack

## Data Engineering

* Python
* PySpark
* Databricks
* Delta Lake
* Databricks SQL

## Machine Learning

* Spark MLlib
* Logistic Regression
* StringIndexer
* OneHotEncoder
* VectorAssembler
* Median Imputation

## Visualization

* Tableau Public

## Development and Version Control

* Git
* GitHub
* Databricks CLI

---

# Repository Structure

```text
construction-permit-delay-intelligence/
│
├── README.md
│
├── notebooks/
│   ├── 02_bronze.py
│   ├── 03_silver.py
│   ├── 04_gold.py
│   ├── 05_eda.py
│   └── 06_modeling.py
│
├── dashboard/
│   └── tableau_dashboard.png
│
├── data/
│   └── README.md
│
└── .gitignore
```

---

# Notebook Organization

## `02_bronze.py`

Responsible for:

* Loading the source dataset
* Inspecting the raw schema
* Creating the Bronze Delta table

Output:

```text
bronze_construction_permits
```

---

## `03_silver.py`

Responsible for:

* Standardizing column names
* Converting dates
* Calculating processing time
* Performing data-quality analysis
* Creating the Silver Delta table

Output:

```text
silver_construction_permits
```

---

## `04_gold.py`

Responsible for:

* Filtering completed permits
* Creating the delayed target
* Creating temporal features
* Creating analytical features
* Persisting the Gold Delta table

Output:

```text
gold_construction_permits
```

---

## `05_eda.py`

Responsible for:

* Descriptive statistics
* Processing-time analysis
* Delay-rate analysis
* Temporal analysis
* Missing-value analysis
* Cohort analysis

---

## `06_modeling.py`

Responsible for:

* Creating the 2025 modeling cohort
* Train/test splitting
* Feature preprocessing
* Logistic Regression training
* Model evaluation
* Confusion-matrix analysis

---

# Reproducibility

The notebooks contain the core Databricks/PySpark workflow used to reproduce the analysis.

The high-level process is:

```text
1. Obtain the DC construction permit dataset
                  |
                  v
2. Load the raw data into Databricks
                  |
                  v
3. Create Bronze Delta table
                  |
                  v
4. Clean and standardize into Silver
                  |
                  v
5. Calculate processing time
                  |
                  v
6. Create Gold analytical dataset
                  |
                  v
7. Perform exploratory analysis
                  |
                  v
8. Select the 2025 modeling cohort
                  |
                  v
9. Train Logistic Regression
                  |
                  v
10. Evaluate on held-out test data
                  |
                  v
11. Export analytical data for Tableau
```

The raw source dataset is not committed to the repository.

This repository contains the analytical workflow and notebooks rather than the source data itself.

---

# Future Improvements

Several extensions could improve the project in a future iteration.

## More Advanced Models

Potential alternatives include:

* Random Forest
* Gradient-Boosted Trees
* XGBoost
* LightGBM
* Neural-network classifiers

These models could be compared against the Logistic Regression baseline.

---

## Better Temporal Validation

A future version could use more sophisticated temporal validation strategies that explicitly account for permit completion timing and censoring.

This would provide a stronger estimate of how the model would behave when applied to genuinely future permits.

---

## Survival Analysis

Because many permits are still in progress and do not yet have final processing times, survival-analysis techniques could provide a more appropriate framework for modeling time-to-completion.

Potential approaches include:

* Kaplan-Meier analysis
* Cox proportional hazards models
* Accelerated failure-time models
* Survival gradient boosting

This would allow incomplete permits to contribute information rather than simply being excluded.

---

## Additional Feature Engineering

Future versions could incorporate:

* More detailed work descriptions
* Permit category information
* Geographic clustering
* Neighborhood-level features
* Historical applicant processing patterns
* Historical contractor processing patterns
* Seasonal effects
* Permit workload indicators
* Application volume by time period

These features could potentially improve predictive performance.

---

## Model Explainability

A future version could incorporate explainability techniques such as:

* Logistic Regression coefficients
* SHAP values
* Feature importance
* Partial dependence analysis

This would help translate model predictions into more actionable insights for permit-processing stakeholders.

---

# Project Status

## Completed

* [x] Public dataset ingestion
* [x] Databricks setup
* [x] Bronze Delta layer
* [x] Silver Delta layer
* [x] Gold analytical layer
* [x] Data-quality analysis
* [x] Processing-time calculation
* [x] Delay-target definition
* [x] Exploratory data analysis
* [x] Temporal cohort analysis
* [x] Logistic Regression baseline
* [x] Model evaluation
* [x] Tableau dashboard development
* [x] GitHub repository preparation

## Current Deliverables

The project currently contains:

* Databricks/PySpark notebooks
* Delta Lake data-engineering workflow
* Exploratory analysis
* Machine-learning baseline
* Model evaluation
* Tableau dashboard
* Project documentation

---

# Conclusion

This project demonstrates an end-to-end approach to analyzing real-world operational data.

Rather than treating machine learning as the entire project, the workflow begins with raw public data and moves through data engineering, quality analysis, feature engineering, exploratory analysis, predictive modeling, and visualization.

The final baseline model achieves a **0.831 ROC-AUC** on held-out 2025 data, while the broader analysis reveals an important limitation in the source data: recent completed permits are subject to an observation-window effect that makes simple year-over-year comparisons misleading.

The project therefore demonstrates both the ability to build a predictive system and the ability to recognize when the underlying data makes a seemingly obvious conclusion unreliable.

That distinction is central to responsible applied machine learning.
