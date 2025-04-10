# Kaggle Survey Salary Classification: Ordinal Logistic Regression

This repository contains an end-to-end machine learning workflow focused on predicting the yearly compensation bucket for data science professionals using the Kaggle 2022 ML & DS Survey dataset. The project centers on an ordinal classification task where the target variable is the respondents’ salary bucket (encoded from the survey). The analysis leverages extensive data cleaning, exploratory data analysis, feature engineering, feature selection, model training, and hyperparameter tuning techniques—all performed in Python on Google Colab.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
  - [Data Cleaning & Missing Value Treatment](#data-cleaning--missing-value-treatment)
  - [Feature Engineering & Encoding](#feature-engineering--encoding)
  - [Exploratory Data Analysis and Feature Selection](#exploratory-data-analysis-and-feature-selection)
  - [Model Implementation](#model-implementation)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Final Model Evaluation](#final-model-evaluation)
- [Results and Insights](#results-and-insights)
- [How to Run the Notebook](#how-to-run-the-notebook)
- [Dependencies and Setup](#dependencies-and-setup)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Project Overview

This project aims to predict salary buckets (ordinal target) of data science professionals by analyzing the 2022 Kaggle ML & DS Survey data. By implementing an ordinal logistic regression model—and experimenting with other techniques such as one-vs-rest (OVR) multiclass classification—the project addresses key challenges in real-world data:
  
- **Handling Missing Values:** Differentiating between truly missing data and “not selected” responses.
- **Data Cleaning and Transformation:** Correcting textual anomalies (e.g., garbage text in education features) and ensuring consistency in data representation.
- **Feature Engineering and Selection:** Utilizing custom encoding for ordinal and nominal data and leveraging Random Forest-based feature selection to reduce dimensionality.
- **Model Training and Hyperparameter Tuning:** Using 10-fold cross-validation and grid search to optimize parameters (including regularization strength, maximum iterations, and class weights) in an effort to improve the model's F1-score.
- **Model Evaluation:** Comparing predictions on training and test sets to diagnose potential issues like underfitting.

---

## Dataset Description

The dataset for this assignment is the cleaned version of the Kaggle Survey 2022 responses stored in `clean_kaggle_data_2022.csv`. Key characteristics include:

- **Data Points:** Over 23,000 participants (original survey), with a subset used after cleaning.
- **Features:** 296 survey questions covering demographics, professional experience, education level, coding and ML experience, and more.
- **Target Variables:**  
  - **Q29_buckets:** The aggregated salary bucket (categorical).  
  - **Q29_Encoded:** Label-encoded salary bucket used as the target for model training.
- **Additional Files:** Auxiliary files (e.g., original responses, methodology) were used to guide the data cleaning and transformation process.

---

## Methodology

### Data Cleaning & Missing Value Treatment

- **Missing Value Analysis:**  
  - A complete analysis is run to count and visualize missing values across all features.
  - Missing values are handled by distinguishing between non-selection (e.g., multiple-choice questions) and genuine missingness.
  
- **Missing Value Treatment:**  
  - **Categorical Features:** Missing entries are filled with a custom label such as “No Selection” or “Unknown.”
  - **Numerical Features:** Missing values are replaced with the median.
  - **Extra Cleaning:** Anomalous textual entries (for example, in education level) are corrected using regex and string replacements.

### Feature Engineering & Encoding

- **Ordinal Encoding:**  
  - Applied to education level, coding experience, age ranges, ML experience, ML spending, team size, and company size.
  - Custom mappings translate categorical grades into integer values.

- **Binary Encoding:**  
  - Features such as student status and academic publishing are converted into binary representations.

- **One-Hot and Label Encoding:**  
  - One-Hot Encoding is used for multi-category features where no natural ordering exists.
  - Label Encoding is applied to features like gender for simple classification requirements.

### Exploratory Data Analysis and Feature Selection

- **Exploratory Visualization:**  
  - Key visualizations include the distribution of compensation buckets, age distribution, and correlation analyses (e.g., boxplots showing the relationship between age and salary).
  
- **Feature Selection:**  
  - Tree-based models like the Random Forest Classifier are trained to generate a feature importance ranking.
  - **SelectFromModel:** Used to extract a subset of features (47 in this case) that are most impactful in predicting the target variable.

### Model Implementation

- **Data Splitting and Scaling:**  
  - The dataset is divided into 80% training and 20% testing samples.
  - Numerical features are scaled using StandardScaler to ensure comparability.
  
- **Ordinal Logistic Regression:**  
  - The classification model is implemented using a one-vs-rest (OVR) strategy.
  - A 10-fold cross-validation approach is applied to assess model accuracy and its stability.

### Hyperparameter Tuning

- **Grid Search:**  
  - Hyperparameters, including the regularization parameter (C), maximum iterations, and class weight configuration, are tuned.
  - Best parameters identified: C = 0.001, max_iter = 100, and class_weight = ‘balanced.’
  
- **Evaluation Metrics:**  
  - Alongside accuracy, F1 score, precision, and recall are computed.
  - The F1 score is especially important given the imbalanced nature of the compensation buckets.

### Final Model Evaluation

- **Model Training:**  
  - The final tuned model is retrained on the entire training set.
  
- **Testing:**  
  - Predictions are made on both the training and test sets.
  - Detailed classification reports are generated.
  
- **Visualization:**  
  - Plots comparing the true salary bucket distributions versus the model’s predictions are generated to assess performance and diagnose potential under/overfitting.

---

## Results and Insights

- **Data Cleaning Impact:**  
  Ensured high data quality by appropriately addressing missing values and inconsistencies, thus preserving valuable information.

- **Feature Selection Outcome:**  
  A reduced set of 47 features proved sufficient to represent the most significant contributors to salary prediction, including age, education level, and respondent country.

- **Model Performance:**  
  Although initial cross-validation yielded an average accuracy of around 37.95%, rigorous hyperparameter tuning improved the F1 score to approximately 0.285.  
  Both training and testing evaluations suggest that the model might be underfitting, indicating further improvements (such as more advanced feature engineering or alternative modeling approaches) are necessary.

---

## How to Run the Notebook

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/kaggle-survey-salary-classification.git
   cd kaggle-survey-salary-classification
   ```

2. **Upload Dataset File:**
   - Ensure `clean_kaggle_data_2022.csv` is in the repository’s root directory.
   - When using Google Colab, use the provided commands to upload the file:
     ```python
     from google.colab import files
     uploaded = files.upload()
     ```

3. **Install Required Libraries:**
   ```bash
   pip install pandas numpy scipy matplotlib seaborn scikit-learn
   ```
---

## Dependencies and Setup

- **Python Version:** 3.x
- **Key Libraries:**
  - `pandas`, `numpy` (data manipulation)
  - `scipy` (statistical analysis)
  - `matplotlib`, `seaborn` (visualizations)
  - `scikit-learn` (machine learning, preprocessing, model selection)

---

## Project Structure

```
kaggle-survey-salary-classification/
├── clean_kaggle_data_2022.csv    # Cleaned dataset
├── survey_analysis.ipynb   # Jupyter Notebook containing code 
├── README.md                     # This README file
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---
