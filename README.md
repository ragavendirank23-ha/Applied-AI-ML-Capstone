# Part 1: Data Preparation and Exploratory Data Analysis (EDA)

## 🏢 1. Dataset Description & Justification
For this project, the **Medical Cost Insurance Dataset** was chosen. It consists of 1,338 records containing individual clinical and personal attributes used to predict medical insurance costs. 
* **Target Variable:** `charges` (Continuous numeric variable representing personal medical costs billed by health insurance).
* **Predictors / Features:** Includes numeric variables (`age`, `bmi`, `children`) and categorical categories (`sex`, `smoker`, `region`).

This dataset provides an ideal canvas for predictive modeling: it exhibits significant feature skewness, has clear multi-variable non-linear interactions, and contains strong categorical driver markers that will act as robust signals for feature selection in Part 2.

---

## 🧹 2. Data Cleaning and Imputation Strategy
During initial auditing, missing data was handled carefully to avoid statistical bias:
* **Null Value Percentage Audit:** Initial inspection checked total missing configurations per feature column to build an isolated modification strategy (`bmi`: 5.01%, `charges`: 2.99%).
* **Skewness-Driven Imputation Choice:** 
  * Features with low skewness distributions (`bmi` with a skew of 0.29) were safely imputed via **Mean** values to maintain spatial consistency.
  * Features exhibiting high skewness metrics (the target `charges` with a skew of 1.52) were strictly imputed via **Median** strategies to defend against the mathematical pulling effects of outliers.
* **Verification:** A terminal validation check using `.isnull().sum()` confirmed that exactly 0 missing null entries remain in the final dataset asset exported to `cleaned_data.csv`.

---

## 📈 3. Statistical Analysis Insights

### Non-Linear Relationship Analysis (Pearson vs. Spearman)
To uncover non-linear structural patterns, we measured standard **Pearson correlation** (which evaluates strictly linear trends) against **Spearman Rank correlation** (which tests for monotonic relationships irrespective of linearity). 
* **Top Identified Discrepancy Pairs:** The largest deviation occurs between `age` and `charges` with an absolute difference of **0.233991**.
* **Interpretation:** Because Spearman's correlation coefficient significantly outscores Pearson's, it indicates a distinct non-linear, monotonic curve structure. An increment in age prompts a continuous change in charges, but not at a constant linear rate. 
* **Feature Selection Guidance:** To accommodate these non-linear properties in Part 2, standard feature engineering or non-linear models (like Tree-Based Regression systems) will be deployed over standard linear models.

### Categorical Driver & Signal Strength Evaluation
Grouping the target variable `charges` across the categorical feature `smoker` yielded highly definitive operational metrics:
* **Implication of Within-Group Variance:** The standard deviation (`std`) inside the smoker group is significantly wide (12,022.07 vs 5,919.83 for non-smokers). This structural variance suggests that while smoking acts as an aggregate catalyst raising base costs, structural attributes inside that group (such as BMI or Age) cause massive personal data spreads.
* **Mean Ratio Predictive Power:** The ratio computed between the highest group mean and lowest group mean is immensely large (**3.67**). This extreme divergence explicitly confirms that the categorical element `smoker` possesses an overwhelming predictive signal. It acts as an absolute key feature driver that the downstream machine learning models must prioritize.

---

## 🎨 4. Visualization Interpretations
All plots have been generated and compiled directly to the repository via automated backend rendering pipeline commands:

1. **`distribution_charges.png` (Distribution Plot):** Shows that the target variable `charges` is severely right-skewed with a heavy tail stretching toward expensive tiers, verifying our decision to utilize a median-imputation defensive posture.
2. **`boxplot_smoker_charges.png` (Box Plot):** Displays a completely distinct, elevated cost box range for smokers compared to non-smokers, confirming our categorical driver signal analysis.
3. **`scatterplot_bmi_charges.png` (Scatter Plot):** Uncovers a massive non-linear interaction effect. Non-smokers exhibit flat costs as BMI increases, whereas smokers present a violent, tiered spike in charges once BMI moves past the critical threshold of 30.
4. **`pairplot_matrix.png` (Pairwise Distribution Matrix):** Illustrates the interconnected dependencies between age, BMI, and charges, highlighted clearly by smoking categories.
5. **`correlation_heatmap.png` (Comparative Correlation Heatmap):** Visually confirms the structural deviation between standard Pearson linear assumptions and Spearman rank orders across your data columns.

## 📊 Part 2: Predictive Modeling & Regularization Insights

### 1. Label Definitions & Target Framework
To evaluate predictive behavior comprehensively, two separate modeling tracks were established from the cleaned dataset:
*   **Regression Target (`y_reg`):** The continuous variable `charges` representing personal medical costs billed by health insurance.
*   **Classification Target (`y_clf`):** A binary indicator derived by binarizing `charges` at its median value ($1$ if charges are strictly greater than the median, $0$ otherwise). This guarantees perfectly balanced baseline class counts ($50\%$ high-cost vs. $50\%$ low-cost), making it a robust diagnostic for threshold adjustments.

---

### 2. Categorical Encoding & Multicollinearity Prevention
The dataset's nominal variables (`sex`, `smoker`, `region`) possess no natural logical or numerical order. 
*   **Approach:** One-hot encoding via `pd.get_dummies(drop_first=True)` was applied to transform these columns into discrete numeric flags.
*   **Justification:** Utilizing label encoding here would mistakenly force a false ordinal relationship (e.g., mapping regions as $0 < 1 < 2 < 3$), deceiving the linear models into treating arbitrary categories as mathematical progression scales. Dropping the first dummy column prevents the "dummy variable trap," eliminating perfect multicollinearity so coefficients remain uniquely identifiable.

---

### 3. Leak-Free Preprocessing Pipeline
To enforce strict experimental validity, data split and scaling were sequenced defensively:
1.  The features and labels were split into training ($80\%$) and testing ($20\%$) subsets using a static `random_state=42`.
2.  A `StandardScaler` was instantiated and fitted **exclusively on the training partition features** (`scaler.fit(X_train)`), then applied to transform both `X_train` and `X_test`.

> ⚠️ **Data Leakage Violation Note:** Fitting a scaler on the entire dataset prior to splitting constitutes severe data leakage. It introduces out-of-sample properties (the global mean and standard deviation of the test set) directly into the training matrix, artificially deflating generalization errors and generating deceptively optimistic validation performance.

---

### 4. Regression Analysis: OLS vs. Ridge Regularization
Both Ordinary Least Squares (OLS) Linear Regression and Ridge Regression ($\alpha=1.0$) were evaluated over the scaled test space. Because the feature matrix has a low dimension relative to sample size without extreme collinearity, both models achieved identical performance metrics:

*   **OLS Linear Regression:** MSE: `22,482,456.88` | $R^2$ Score: `0.7830`
*   **Ridge Regression:** MSE: `22,482,456.88` | $R^2$ Score: `0.7830`

#### Model Coefficients Interpretation:
*   **`smoker_yes` (+9,573.00):** By an overwhelming margin, smoking status is the single strongest driver of medical charges.
*   **`age` (+3,623.51) & `bmi` (+2,059.03):** Incremental increases in age and body mass index yield clear linear expansions in billed insurance costs.
*   **`children` (+518.23):** Dependent counts exert a positive but noticeably smaller pressure on expenditure.
*   **Demographics (`sex_male`, `region_*`):** Features like gender and geographical routing capture minor variation, remaining close to zero.

---

### 5. Binary Classification & Threshold Sensitivity Analysis
A baseline Logistic Regression model with default inverse regularization strength ($C=1.0$) and `class_weight='balanced'` was trained to predict high-cost patients. 

#### Decision-Threshold Sensitivity Table
Modifying the classification probability threshold shifts the structural behavior of our predictions significantly:

| Threshold | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **0.30** | 0.7303 | 0.9924 | 0.8414 |
| **0.40** | 0.8113 | 0.9847 | 0.8897 |
| **0.50 (Baseline)** | 0.8714 | 0.9313 | 0.9004 |
| **0.60** | 0.9412 | 0.8550 | 0.8961 |
| **0.70** | 0.9633 | 0.8015 | 0.8750 |

#### Strategic Trade-offs:
*   **Optimizing for Recall (Low Threshold = 0.30):** Catching $99.2\%$ of high-cost cases minimises the risk of missing high-risk individuals, though it forces insurance staff to sift through more false positives (lower precision).
*   **Optimizing for Precision (High Threshold = 0.70):** Ensures that $96.3\%$ of flagged individuals are truly high-cost cases, making it ideal for strict budget allocation where resources cannot be wasted on false alarms.

---

### 6. Regularization Experiment (Baseline C=1.0 vs. Strong C=0.01)
To explicitly control model complexity, a heavily regularized Logistic Regression variant ($C=0.01$) was introduced. 
*   **Baseline ($C=1.0$):** Reached an outstanding Overall Accuracy of `0.90` and an Area Under the ROC Curve (AUC) of roughly `0.96`.
*   **Strong Regularization ($C=0.01$):** Accuracy held steady at `0.89` with a minimal reduction in structural capability. The aggressive penalty compacted feature weights but maintained strong descriptive viability due to the highly distinct signal of features like `smoker_yes`.

---

### 7. Statistical Significance via Bootstrap Confidence Intervals
To prove whether the regularized model significantly underperformed compared to the baseline, non-parametric bootstrapping was executed over $500$ resamples to track the difference in area under the curve ($\Delta \text{AUC} = \text{AUC}_{\text{Baseline}} - \text{AUC}_{\text{Regularized}}$):

*   **Mean AUC Difference:** `0.0011`
*   **95% Percentile Confidence Interval:** `[-0.0012, 0.0034]`

#### Scientific Conclusion:
Because the **95% Confidence Interval contains zero** (`-0.0012` to `0.0034`), the minor performance difference between models is **statistically insignificant**. We can confidently implement the simpler, highly regularized $C=0.01$ model to guarantee lighter computational complexity without sacrificing predictive quality.
## 🚀 Part 3: Model Deployment & Production Documentation

### 1. Deployment Artifact Integrity
The trained preprocessing pipeline and baseline classification model have been serialized into a unified deployment file to ensure production environment consistency:
*   **File Name:** `trained_insurance_model.pkl`
*   **Path in Repository:** `/trained_insurance_model.pkl`
*   **Contents:** A serialized Python dictionary housing the fitted `StandardScaler` instance, the trained baseline `LogisticRegression` model, and the explicit structural column order of the one-hot encoded feature matrix.

---

### 2. Production Loading & Inference Guide
To integrate this model into a live production tracking system or an API endpoint (such as FastAPI or Flask), use the following Python verification script to load the file and generate real-time predictions:

```python
import pickle
import pandas as pd
import numpy as np

# Step 1: Load the deployment artifact safely
artifact_path = 'trained_insurance_model.pkl'
with open(artifact_path, 'rb') as f:
    artifact = pickle.load(f)

scaler = artifact['scaler']
model = artifact['model']
expected_features = artifact['features']

print("🚀 Production Model and Scaler loaded successfully!")

# Step 2: Define a new, unseen sample profile for prediction
# Scenario: A 45-year-old male, BMI of 28.5, 2 children, who is a smoker living in the southwest
raw_sample = {
    'age': 45,
    'bmi': 28.5,
    'children': 2,
    'sex_male': 1,
    'smoker_yes': 1,
    'region_northwest': 0,
    'region_southeast': 0,
    'region_southwest': 1
}

# Convert sample to DataFrame and match production column structure exactly
input_df = pd.DataFrame([raw_sample])[expected_features]

# Step 3: Scale features using the production scaler to eliminate data mismatch
scaled_input = scaler.transform(input_df)

# Step 4: Generate prediction probability and binary class choice
prob_high_cost = model.predict_proba(scaled_input)[0][1]
predicted_class = model.predict(scaled_input)[0]

print(f"Analysis Results:")
print(f"-> Probability of exceeding median medical charges: {prob_high_cost * 100:.2f}%")
print(f"-> Final Production Classification: {'High Cost Patient (1)' if predicted_class == 1 else 'Low Cost Patient (0)'}")

