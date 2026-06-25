# Product Intelligence and Customer Review Analytics

### KPI-Based Product Segmentation and Explainable Sentiment Classification Using Amazon Beauty Reviews

An end-to-end product analytics and machine-learning project that transforms large-scale customer review data into product-level KPIs, interpretable product segments, risk-prioritization insights, and sentiment predictions.

---

## Project Overview

Online marketplaces receive more customer feedback than product managers can evaluate manually. Although individual reviews provide valuable information, decision-makers require scalable methods for summarizing product performance, identifying unusual patterns, and prioritizing products for further investigation.

This project develops a **Product Intelligence and Customer Review Analytics framework** using the Amazon Reviews 2023 All Beauty dataset.

The workflow combines:

* exploratory data analysis;
* feature engineering;
* product-level KPI development;
* product risk prioritization;
* correlation analysis and KPI validation;
* K-Means product segmentation;
* Principal Component Analysis for visualization;
* TF-IDF text vectorization;
* Logistic Regression sentiment classification;
* interpretable model analysis;
* managerial recommendations.

The objective is not only to build machine-learning models, but also to demonstrate how review data can support evidence-based product-management decisions.

---

## Business Problem

A product manager working with a large marketplace may need to answer questions such as:

* Which products receive the most concerning customer feedback?
* Which products consistently generate positive customer experiences?
* Are some product KPIs measuring the same underlying behavior?
* Can products be divided into meaningful behavioral segments?
* Can customer sentiment be identified automatically from review text?
* Which words and phrases most strongly influence sentiment predictions?
* How can analytical findings be converted into practical management actions?

This project develops a structured framework for investigating these questions.

---

## Research Questions

The project addresses the following questions:

1. What patterns exist in Amazon Beauty customer ratings and review behavior?
2. Which KPIs can summarize product-level customer feedback?
3. Why can average product ratings be misleading when review volume is low?
4. Are any product KPIs strongly correlated or redundant?
5. Can unsupervised machine learning identify meaningful product segments?
6. Can review sentiment be classified using review text alone?
7. Which textual features contribute most strongly to positive and negative predictions?
8. How can these findings support product-management decisions?

---

## Dataset

This project uses the **Amazon Reviews 2023** dataset published by the McAuley Lab.

**Selected category:** `All_Beauty`

The review file contains approximately:

* 701,528 customer-review records;
* 112,565 unique parent-product identifiers;
* 631,986 unique reviewers;
* 10 original variables.

The available variables include:

| Variable            | Description                                    |
| ------------------- | ---------------------------------------------- |
| `rating`            | Customer star rating from 1 to 5               |
| `title`             | Review title                                   |
| `text`              | Full review text                               |
| `images`            | Images attached to the review                  |
| `asin`              | Amazon item identifier                         |
| `parent_asin`       | Parent-product identifier used for aggregation |
| `user_id`           | Anonymized reviewer identifier                 |
| `timestamp`         | Review timestamp                               |
| `helpful_vote`      | Number of helpful votes                        |
| `verified_purchase` | Whether Amazon marked the purchase as verified |

### Important Scope Note

This is a **customer-review dataset**, not a complete sales dataset.

Therefore:

* review count is interpreted as feedback volume, not exact sales volume;
* verified-purchase rate is treated as a review characteristic, not proof of product quality;
* risk scores identify products for further investigation and do not prove that a product is defective;
* relationships found in the data represent associations and should not automatically be interpreted as causal effects.

---

## Downloading the Data

The dataset is not included in this repository because of its size and to preserve attribution to the original source.

1. Visit the official [Amazon Reviews 2023 dataset page](https://amazon-reviews-2023.github.io/main.html).
2. Locate the **All Beauty** category.
3. Download the review file.
4. Save the compressed file with the following name:

```text
All_Beauty.jsonl.gz
```

5. Place it in the same directory as the notebook, or update `DATA_PATH` inside the notebook.

The notebook expects the following default path:

```python
DATA_PATH = Path("All_Beauty.jsonl.gz")
```

---

## Analytical Workflow

The project follows this general process:

```text
Business Problem Definition
            ↓
Dataset Loading and Validation
            ↓
Data Preparation and Feature Engineering
            ↓
Exploratory Data Analysis
            ↓
Product-Level KPI Development
            ↓
Product Reliability Filtering
            ↓
Risk Prioritization
            ↓
Correlation Analysis and KPI Validation
            ↓
K-Means Product Segmentation
            ↓
PCA Cluster Visualization
            ↓
TF-IDF Text Vectorization
            ↓
Logistic Regression Classification
            ↓
Model Evaluation and Interpretation
            ↓
Managerial Recommendations
```

---

## 1. Data Preparation and Feature Engineering

The raw review data are prepared before advanced analysis.

Engineered review-level features include:

### Review Length

The number of characters in each customer review:

```python
reviews["review_length"] = reviews["text"].fillna("").str.len()
```

Review length is used as a proxy for the amount of written customer engagement. It does not independently indicate whether a review is positive or negative.

### Negative Review Indicator

Reviews with ratings of one or two stars are classified as negative:

```python
reviews["negative_review"] = (
    reviews["rating"] <= 2
).astype(int)
```

This feature supports calculation of the product-level negative-review rate.

---

## 2. Exploratory Data Analysis

The exploratory analysis examines:

* rating distribution;
* review-length distribution;
* helpful-vote behavior;
* verified-purchase distribution;
* review-volume characteristics.

EDA is used to identify patterns, skewness, outliers, and potential limitations before statistical modeling.

One important finding is that product review volume is highly uneven. Many products have very few reviews, while a much smaller number receive substantial feedback.

---

## 3. Product-Level KPI Framework

Individual reviews are aggregated by `parent_asin`, producing one analytical row per product.

The following product-level indicators are developed:

| KPI                    | Interpretation                               |
| ---------------------- | -------------------------------------------- |
| Average Rating         | Overall rating-based customer satisfaction   |
| Review Count           | Volume of available customer feedback        |
| Negative Review Rate   | Proportion of reviews rated one or two stars |
| Verified Purchase Rate | Proportion marked as verified purchases      |
| Average Helpful Votes  | Average helpfulness feedback received        |
| Average Review Length  | Average amount of written feedback           |

The aggregation process converts review analytics into product-level intelligence.

---

## 4. Statistical Reliability Filter

The full dataset contains approximately 112,565 unique parent products. However, many products are supported by only one or two reviews.

An average rating based on one review is less stable than an average based on hundreds of reviews.

For advanced product analysis, this project retains products with at least:

```text
20 reviews
```

This produces a more reliable subset of approximately:

```text
6,044 products
```

The threshold is an analytical choice rather than a universal statistical rule. Products below the threshold are not considered unimportant; they are excluded from selected analyses because their KPI estimates may be unstable.

---

## 5. Product Risk Prioritization

An exploratory Product Risk Score is created to combine multiple indicators into one prioritization measure.

The initial score uses:

* inverse average rating;
* negative-review rate;
* inverse verified-purchase rate.

```python
risk_score = (
    (1 - avg_rating / 5) * 40
    + negative_review_rate * 40
    + (1 - verified_purchase_rate) * 20
)
```

### Interpretation

A higher value indicates a more concerning combination of product-review characteristics.

The score is intended to support prioritization, not to make a final judgment about product quality.

### Limitation

The weighting structure is manually defined and has not been optimized against a validated business outcome. The score is therefore treated as an exploratory decision-support prototype.

---

## 6. KPI Validation and Correlation Analysis

Correlation analysis is used to examine whether the KPIs contribute distinct information.

The strongest relationship was found between:

```text
Average Rating ↔ Negative Review Rate
```

with a Pearson correlation of approximately:

```text
r = -0.976
```

This extremely strong inverse relationship indicates substantial redundancy.

Products with higher average ratings generally have lower negative-review rates, while products with lower ratings generally have higher negative-review rates.

### Implication

Because the initial risk score includes both variables, it may partially count customer dissatisfaction twice.

This finding demonstrates why KPI systems should be statistically validated rather than accepted solely because they appear intuitively reasonable.

Review count and verified-purchase rate showed much weaker relationships with average rating.

---

## 7. Product Segmentation with K-Means

K-Means clustering is applied to identify groups of products with similar KPI behavior.

### Features Used

* Average Rating
* Review Count
* Verified Purchase Rate
* Average Helpful Votes
* Average Review Length

Highly skewed variables are log-transformed before clustering:

```python
np.log1p()
```

All clustering features are then standardized using:

```python
StandardScaler()
```

Standardization prevents variables with larger numerical ranges from dominating distance calculations.

### Cluster Interpretation

The model identified four product segments:

| Cluster | Business Label         | Main Characteristics                                                             |
| ------: | ---------------------- | -------------------------------------------------------------------------------- |
|       0 | Popular Products       | Substantially higher review volume and generally favorable ratings               |
|       1 | High-Risk Products     | Lowest ratings, highest negative-review rate, and highest exploratory risk score |
|       2 | Trust Anomaly Products | Relatively favorable ratings but unusually low verified-purchase rate            |
|       3 | Customer Favorites     | Highest ratings, lowest negative-review rate, and lowest risk score              |

The descriptive names were assigned **after** examining the cluster profiles. K-Means itself produced only numerical cluster labels.

### Cluster Sizes

| Segment                | Product Count |
| ---------------------- | ------------: |
| Popular Products       |         1,270 |
| High-Risk Products     |         1,357 |
| Trust Anomaly Products |           628 |
| Customer Favorites     |         2,789 |

Cluster numbering is specific to the fitted model and may change if the data, random seed, preprocessing, or scikit-learn version changes.

---

## 8. Principal Component Analysis

K-Means operates in a multidimensional feature space that cannot be directly displayed on a two-dimensional chart.

Principal Component Analysis is therefore used to project the standardized KPI data onto two components for visualization.

The first two principal components explain approximately:

| Component             | Explained Variance |
| --------------------- | -----------------: |
| Principal Component 1 |             30.99% |
| Principal Component 2 |             23.30% |
| Combined              |             54.28% |

PCA did **not** create the clusters. It was applied after clustering to visualize the existing K-Means assignments.

Because two components preserve only about 54.28% of total variance, the PCA chart is an informative approximation rather than a complete representation of all five original features.

---

## 9. NLP Sentiment Classification

The project also develops a supervised NLP model that predicts whether a customer review is positive or negative using its text.

### Sentiment Labels

Ratings are converted into binary labels:

```text
4–5 stars → Positive
1–2 stars → Negative
3 stars   → Excluded as ambiguous
```

The sentiment target is therefore derived from customer ratings.

This means the model learns to predict rating-based sentiment from text; it is not validated against independently hand-labeled emotions.

### Text Input

The model combines:

```text
Review Title + Review Text
```

Reviews containing fewer than 10 characters are excluded from the modeling sample.

A reproducible sample of 100,000 reviews is used to reduce computational requirements.

---

## 10. TF-IDF Vectorization

Machine-learning algorithms cannot directly process raw text.

TF-IDF—Term Frequency–Inverse Document Frequency—converts review text into numerical features.

It gives greater weight to words or phrases that:

* are important within a particular review;
* are not excessively common throughout all reviews.

The vectorizer uses:

```python
TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)
```

This configuration includes:

* up to 10,000 text features;
* removal of common English stop words;
* single words;
* two-word phrases.

The vectorizer is fitted only on the training data to avoid test-set information leakage.

---

## 11. Logistic Regression

Logistic Regression is used as an interpretable binary classification model.

The processing pipeline is:

```text
Review Text
     ↓
TF-IDF Numerical Features
     ↓
Logistic Regression
     ↓
Positive or Negative Prediction
```

Although its name includes “regression,” Logistic Regression is commonly used for classification problems.

The model estimates the probability that each review belongs to the positive class.

---

## 12. Model Performance

The model achieved approximately:

```text
Accuracy: 93.5%
```

Detailed test-set results:

| Class            | Precision | Recall | F1-Score | Test Examples |
| ---------------- | --------: | -----: | -------: | ------------: |
| Negative         |      0.90 |   0.80 |     0.85 |         4,520 |
| Positive         |      0.94 |   0.97 |     0.96 |        15,480 |
| Weighted Average |      0.93 |   0.94 |     0.93 |        20,000 |

### Confusion Matrix

```text
                    Predicted Negative    Predicted Positive
Actual Negative            3,614                  906
Actual Positive              393               15,087
```

### Interpretation

The model performs particularly well on positive reviews.

Negative-review recall is lower at approximately 0.80. This means that around 20% of negative reviews in the test set were incorrectly classified as positive.

For a product-monitoring system, this is an important limitation because missed negative reviews could conceal emerging customer dissatisfaction.

A future production model may therefore prioritize negative-class recall instead of optimizing accuracy alone.

---

## 13. Explainable Sentiment Classification

Logistic Regression assigns a coefficient to every TF-IDF feature.

These coefficients provide a relatively interpretable view of which words and phrases influence predictions.

### Terms Associated with Positive Reviews

Examples include:

```text
excellent
fantastic
highly
easy
favorite
loved
wonderful
beautiful
awesome
amazing
perfect
great
```

### Terms Associated with Negative Reviews

Examples include:

```text
disappointed
return
terrible
horrible
poor
waste
broken
unfortunately
useless
cheap
doesn't work
don't buy
```

These terms indicate that the model learned recognizable customer-sentiment patterns rather than relying on an entirely opaque decision process.

The coefficients show statistical associations with the predicted classes; they should not be interpreted as universal definitions of sentiment in every context.

---

## 14. Key Findings

The main findings are:

* Review data can be transformed into a structured product-level KPI framework.
* Product ratings supported by very low review counts should be interpreted cautiously.
* Average rating and negative-review rate are strongly redundant.
* The initial risk score may overweight dissatisfaction by including both correlated variables.
* Four distinct product segments were identified through K-Means clustering.
* One segment showed substantially higher dissatisfaction and risk.
* Another segment displayed unusually low verified-purchase rates despite relatively favorable ratings.
* Review text contains strong predictive information about rating-based sentiment.
* The sentiment classifier achieved approximately 93.5% accuracy.
* Negative-class recall remains an important area for improvement.
* Logistic Regression coefficients provide an interpretable view of influential language patterns.

---

## 15. Managerial Recommendations

### High-Risk Products

Products within the high-risk segment should be prioritized for:

* detailed complaint analysis;
* supplier or quality review;
* product-description verification;
* packaging and fulfillment investigation;
* customer-service follow-up.

### Customer Favorites

Strong-performing products may support:

* promotional campaigns;
* bundling strategies;
* product-line expansion;
* analysis of repeatable best practices.

### Popular Products

Highly reviewed products deserve continued monitoring because problems affecting these products may influence a larger volume of customers.

Review count should still be interpreted as feedback volume rather than exact sales demand.

### Trust Anomaly Products

Products with unusually low verified-purchase rates may warrant additional review-monitoring analysis.

A low verified-purchase rate does not independently prove review manipulation or fraud.

### NLP Monitoring

The text-classification pipeline could support automated review triage by:

* flagging likely negative reviews;
* detecting emerging dissatisfaction;
* prioritizing reviews for manual inspection;
* monitoring changes in customer language over time.

---

## 16. Limitations

This project has several limitations:

* Reviews are not equivalent to complete transaction or sales records.
* Review count is not a direct measure of product demand.
* Product metadata such as product title, brand, price, and category were not included in the current analysis.
* The 20-review threshold is a practical analytical choice.
* The risk-score weights were manually specified.
* Average rating and negative-review rate are highly correlated.
* The selected number of clusters is partly based on interpretability and should be compared with formal validation metrics.
* K-Means assumes distance-based, approximately spherical clusters.
* PCA preserves only part of the total variance.
* Sentiment labels were generated from ratings rather than independent human annotation.
* Three-star reviews were excluded.
* The NLP sample is class-imbalanced, with more positive than negative reviews.
* TF-IDF does not fully understand context, sarcasm, or complex language.
* Association does not establish causation.

---

## 17. Future Work

Potential extensions include:

* using silhouette analysis and the elbow method to compare cluster counts;
* testing hierarchical clustering, DBSCAN, or Gaussian Mixture Models;
* integrating Amazon product metadata;
* replacing manual risk-score weights with statistically optimized weights;
* analyzing rating and sentiment trends over time;
* developing product-level early-warning indicators;
* testing class weighting or resampling to improve negative-review recall;
* comparing Logistic Regression with Support Vector Machines or tree-based models;
* experimenting with word embeddings and transformer models such as BERT;
* using SHAP or LIME for additional model explanations;
* developing an interactive dashboard with Power BI, Tableau, Plotly, or Streamlit;
* building an automated monitoring pipeline for newly arriving reviews.

---

## 18. Technologies Used

### Programming and Analysis

* Python
* pandas
* NumPy

### Visualization

* Matplotlib

### Machine Learning

* scikit-learn
* K-Means Clustering
* Principal Component Analysis
* Logistic Regression

### Natural Language Processing

* TF-IDF Vectorization
* Unigram and bigram text features

### Development Environment

* Jupyter Notebook
* Anaconda Notebooks

---

## Repository Structure

```text
product-intelligence-customer-review-analytics/
│
├── README.md
├── Product_Intelligence_Customer_Review_Analytics.ipynb
├── requirements.txt
├── .gitignore
└── images/
    ├── executive_dashboard.png
    ├── correlation_heatmap.png
    ├── product_clusters_pca.png
    └── risk_score_distribution.png
```

The dataset is downloaded separately and is not committed to the repository.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/product-intelligence-customer-review-analytics.git
cd product-intelligence-customer-review-analytics
```

Create an optional virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Download `All_Beauty.jsonl.gz` from the official dataset page and place it in the project directory.

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
Product_Intelligence_Customer_Review_Analytics.ipynb
```

Then select:

```text
Kernel → Restart Kernel and Run All Cells
```

---

## Requirements

The project uses the following main libraries:

```text
numpy
pandas
matplotlib
scikit-learn
jupyter
```

Exact versions can be recorded in `requirements.txt` after confirming the final execution environment.

---

## Reproducibility

Randomized operations use:

```python
RANDOM_STATE = 42
```

This is applied to:

* review sampling;
* train/test splitting;
* K-Means initialization.

The notebook is designed to run sequentially from top to bottom.

Results may vary slightly across software versions, particularly K-Means cluster numbering and numerical optimization results.

---

## Academic and Portfolio Purpose

This project was developed as an independent Data Analysis initiative to demonstrate practical skills in:

* product analytics;
* business intelligence;
* KPI design;
* statistical reasoning;
* data preparation;
* unsupervised machine learning;
* natural language processing;
* explainable classification;
* managerial interpretation.

The focus is not only model performance, but also the connection between technical methods and business decision-making.

---

## Author

**Daniyal Hemmati**

Affiliated with Management Information Systems Department

daniyal.hemmati@std.medipol.edu.tr

---

## Acknowledgment

The dataset was created and published by the McAuley Lab as part of the Amazon Reviews 2023 collection.

Please consult the official dataset page for citation information, licensing details, and usage conditions.
