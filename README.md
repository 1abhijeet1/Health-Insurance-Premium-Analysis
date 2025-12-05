# Health Insurance Premium Prediction

This project analyzes a health insurance dataset to understand the factors influencing medical costs and builds predictive models for insurance premiums. It involves comprehensive exploratory data analysis (EDA), statistical testing, feature engineering (including PCA), and Linear Regression modeling.

## Dataset

The dataset used is the [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance) from Kaggle. It includes the following attributes for 1,338 beneficiaries:

* **age**: Age of primary beneficiary.
* **sex**: Insurance contractor gender (female, male).
* **bmi**: Body mass index, providing an understanding of body weights that are relatively high or low relative to height.
* **children**: Number of children covered by health insurance / Number of dependents.
* **smoker**: Smoking status (yes, no).
* **region**: The beneficiary's residential area in the US (northeast, southeast, southwest, northwest).
* **charges**: Individual medical costs billed by health insurance (Target Variable).

## Project Workflow

### 1. Data Preparation & Preprocessing
* **Setup**: Configures Kaggle API to download and extract the dataset.
* **Encoding**:
    * Binary encoding for `sex` (Female=0, Male=1) and `smoker` (No=0, Yes=1).
    * One-hot encoding for the `region` variable.
* **Binning**: Creates an `age_group` category to analyze trends across different age ranges.

### 2. Exploratory Data Analysis (EDA)
* **Correlation Matrix**: Visualizes relationships between numerical features. Smoking status is identified as having the strongest correlation with medical charges.
* **Visualizations**:
    * Scatterplots (e.g., BMI vs. Charges) to observe trends.
    * Bar charts comparing mean charges across different categories (number of children, smoker status, region).
    * Histograms and KDE plots to examine the distribution of variables like BMI and Charges.
* **Statistical Testing**: Performs an ANOVA test to determine if the mean charges differ significantly across different regions.
* **Outlier Detection**: Identifies outliers in `bmi` and `charges` using the Interquartile Range (IQR) method.

### 3. Feature Engineering
* **Log Transformation**: Applies a log transformation to the `charges` variable (`log_charges`) to address right-skewness and improve model performance.
* **Scaling**: Standardizes continuous variables (`age`, `bmi`) using `StandardScaler`.
* **Principal Component Analysis (PCA)**: Reduces the dimensionality of `age` and `bmi` into a single component (`age_bmi_pca`) to potentially handle multicollinearity or capture combined variance.

### 4. Modeling
The project uses **Ordinary Least Squares (OLS)** regression from the `statsmodels` library to quantify relationships.

* **Feature Selection**: Multiple models are likely iterated upon. One key model displayed includes:
    * `age_bmi_pca` (Combined Age & BMI feature)
    * `children`
    * `smoker`
    * `region` (Northwest, Southeast, Southwest)
* **Evaluation**: The model summary provides key metrics:
    * **R-squared**: Indicates the proportion of variance in charges explained by the model.
    * **P-values**: Used to determine the statistical significance of each feature (e.g., `smoker` having a p-value of 0.000).

## Results Highlights
* **Smoking** is the most significant predictor of higher insurance charges.
* **BMI** has a distinct relationship with charges, often showing two clusters (smokers vs. non-smokers).
* The final model utilizes a PCA-transformed feature for Age and BMI to predict premiums.

## Requirements
To run this notebook, you will need the following Python libraries:
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `scipy`
* `statsmodels`

## Usage
1.  Ensure you have the necessary `kaggle.json` API token if you intend to re-download the data directly.
2.  Install dependencies via `pip install -r requirements.txt` (or individually).
3.  Run the Jupyter Notebook `Health Insurance Premium Prediction.ipynb` to execute the code cells.
