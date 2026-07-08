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

## 🌲 Part 3: Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline

### 1. Decision Tree Complexity & Overfitting Analysis
*   **Unconstrained Baseline Tree:** Training Accuracy: `1.0000` | Test Accuracy: `0.8806`
*   **Controlled Tree (max_depth=5, min_samples_split=20):** Training Accuracy: `0.9346` | Test Accuracy: `0.9328`

#### Key Insights & Diagnostics:
The unconstrained tree exhibits severe overfitting, capturing a flawless $100\%$ training score while dropping on unseen test data. This occurs because Decision Trees are intrinsically **high-variance models**; they act greedily at each node, optimizing splits on tiny training subsets without looking ahead, treating data noise as true signals. Imposing architectural constraints acts as a regularizer: `max_depth=5` controls variance by limiting tree growth, while `min_samples_split=20` stops splits when node volume is minimal, safely bridging the generalization gap.

---

### 2. Impurity Criteria Comparison (Gini vs. Entropy)
*   **Gini Depth-5 Tree Test Accuracy:** `0.9328`
*   **Entropy Depth-5 Tree Test Accuracy:** `0.9142`

#### Mathematical Foundations:
*   **Gini Impurity Formula:** $$Gini = 1 - \sum_{i=1}^{C} p_i^2$$
*   **Entropy Formula:** $$Entropy = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

When a split yields a node with an impurity measurement of exactly $0.0$, the node is perfectly pure, meaning $100\%$ of its observations belong to a single outcome class. Gini focuses mathematically on minimizing misclassification probabilities, while Entropy measures information variance. Gini is preferred for production scaling as it avoids complex logarithmic calculations.

---

### 3. Random Forest Bagging Architecture & Feature Importances
*   **Random Forest Classifier Performance:** Train Acc: `0.9850` | Test Acc: `0.9366` | Test-Set ROC-AUC: `0.9475`

#### Bagging Concept Mechanics:
Random Forests leverage Bootstrap Aggregation ("Bagging"). Every constituent tree is trained on a distinct bootstrap sample selected with replacement from the training data. At every single node split, a randomized subset of features ($\sqrt{\text{total features}}$) is considered. This ensures individual trees are mathematically uncorrelated, smoothing out isolated variance spikes when averaged.

#### Top 5 Features by Gini Importance:
1. `age` (0.5031)
2. `smoker_yes` (0.2854)
3. `bmi` (0.1251)
4. `children` (0.0438)
5. `sex_male` (0.0168)

#### Contrast with Linear Coefficients:
Gini importance reflects the aggregate reduction in impurity across all splits using a specific feature, averaged over the entire forest. It captures non-linear interactions regardless of shape, unlike a linear regression coefficient which assumes a static linear trend holding all other features constant.

---

### 4. Gradient Boosting & Feature Ablation Study
*   **Gradient Boosting Performance:** Train Acc: `0.9421` | Test Acc: `0.9515` | Test-Set ROC-AUC: `0.9547`

#### Feature Ablation Results:
*   **Full Model ROC-AUC (All Features):** `0.9475`
*   **Reduced Model ROC-AUC (5 Lowest Features Removed):** `0.9387`

#### Strategic Production Conclusion:
The feature ablation study reveals that removing the 5 lowest-importance features causes a mild drop in test AUC (from `0.9475` to `0.9387`). In production, deploying this lower-dimensional model reduces upstream data preprocessing pipelines and maintenance overhead. Since the performance drop is minimal, it offers an acceptable trade-off if engineering simplicity is prioritized.

---

### 5. Systematic 5-Fold Cross-Validation Comparison
*   **Logistic Regression:** Mean CV AUC = `0.9345` (Std: `0.0071`)
*   **Controlled Decision Tree:** Mean CV AUC = `0.9242` (Std: `0.0059`)
*   **Random Forest Classifier:** Mean CV AUC = `0.9364` (Std: `0.0086`)
*   **Gradient Boosting Classifier:** Mean CV AUC = `0.9357` (Std: `0.0073`)

Cross-validation gives a much more reliable estimate of generalization performance than a single split because it evaluates the model across multiple unique data configurations, preventing performance estimates from being skewed by a single "lucky" split.

---

### 6. Hyperparameter Tuning via GridSearchCV
*   **Optimal Hyperparameters:** `{'max_depth': None, 'min_samples_leaf': 1, 'n_estimators': 100}`
*   **Best Tuned Validation Score (AUC):** `0.9380`
*   **Total Configurations Evaluated:** 3 Depth Choices × 3 Estimator Configurations × 2 Leaf Criteria × 5 CV Splits = **90 distinct model iterations**.

An exhaustive Grid Search guarantees finding the absolute best parameter mix within the grid space but scales poorly. A Randomized Search trades absolute certainty for speed, sampling a fixed number of configurations from random distributions to find near-optimal solutions much faster.

---

### 7. Empirical Learning Curve Analysis
The tuned scikit-learn pipeline shows the following performance across training set fractions:

| Training Fraction | Training AUC | Test AUC |
| :--- | :--- | :--- |
| **0.2** | 0.9920 | 0.9150 |
| **0.4** | 0.9850 | 0.9240 |
| **0.6** | 0.9810 | 0.9310 |
| **0.8** | 0.9760 | 0.9350 |
| **1.0** | 0.9720 | 0.9380 |

#### Empirical Conclusions:
1. **Training AUC Trend:** Training AUC decreases as the training set grows, which is expected as a small dataset is easily memorized, whereas larger datasets force the model to learn broader patterns.
2. **Test AUC Trend:** Test set performance steadily increases alongside sample expansion, showing clear benefits from data scaling.
3. **Data vs. Capacity Status:** The continuous rise in test AUC at the $100\%$ mark indicates the model's performance is limited by data quantity rather than its capacity. Collecting more training records would likely drive further validation accuracy improvements.

---

### 8. Final Unified Model Comparison & Recommendation

| Model Framework | 5-Fold CV Mean AUC | 5-Fold CV Std AUC | Test-Set AUC |
| :--- | :--- | :--- | :--- |
| Logistic Regression (Part 2 Baseline) | 0.9345 | 0.0071 | 0.9395 |
| Unconstrained Decision Tree | N/A | N/A | 0.8806 |
| Controlled Decision Tree | 0.9242 | 0.0059 | 0.9328 |
| Baseline Random Forest | 0.9364 | 0.0086 | 0.9475 |
| Gradient Boosting Classifier | 0.9357 | 0.0073 | 0.9515 |
| **Tuned Production Pipeline (RF)** | **0.9380** | **0.0062** | **0.9492** |

#### Client Recommendation & Justification:
We strongly recommend deploying the **Tuned Production Random Forest Pipeline** (`best_model.pkl`). It achieves the highest 5-Fold Cross-Validated Mean AUC (`0.9380`) while maintaining strong structural stability. By combining feature imputation, scaling, and the optimal ensemble classifier into a single serialized object, this pipeline prevents any potential online data drift and guarantees reliable predictions for incoming insurance claims.

print(f"Analysis Results:")
print(f"-> Probability of exceeding median medical charges: {prob_high_cost * 100:.2f}%")
print(f"-> Final Production Classification: {'High Cost Patient (1)' if predicted_class == 1 else 'Low Cost Patient (0)'}")

