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
