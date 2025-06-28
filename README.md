# Customer Churn Prediction Using Machine Learning

## Table of Contents

1. [Project Overview](https://github.com/tamaxhh/Customer-Churn-Prediction?tab=readme-ov-file#1-project-overview)
2. [Problem Statement / Motivation](https://github.com/tamaxhh/Customer-Churn-Prediction?tab=readme-ov-file#2-problem-statement--motivation)
3. [Dataset Description](https://github.com/tamaxhh/Customer-Churn-Prediction?tab=readme-ov-file#3-dataset-description)
4. [Goals & Objectives](https://github.com/tamaxhh/Customer-Churn-Prediction?tab=readme-ov-file#4-goals--objectives)
5. [Technologies Used](https://github.com/tamaxhh/Customer-Churn-Prediction?tab=readme-ov-file#5-technologies-used)
6. [Installation & Setup](https://github.com/tamaxhh/Customer-Churn-Prediction?tab=readme-ov-file#6-installation--setup)
7. [Project Structure](https://github.com/tamaxhh/Customer-Churn-Prediction?tab=readme-ov-file#7-project-structure)
8. [Methodology](https://github.com/tamaxhh/Customer-Churn-Prediction?tab=readme-ov-file#8-methodology)
    - [Data Loading & Initial Exploration](https://github.com/tamaxhh/Customer-Churn-Prediction?tab=readme-ov-file#data-loading--initial-exploration)
    - [Data Preprocessing](https://github.com/tamaxhh/Customer-Churn-Prediction?tab=readme-ov-file#data-preprocessing)
    - [Exploratory Data Analysis (EDA) Insights](https://github.com/tamaxhh/Customer-Churn-Prediction?tab=readme-ov-file#exploratory-data-analysis-eda-insights)
    - [Feature Engineering](https://github.com/tamaxhh/Customer-Churn-Prediction?tab=readme-ov-file#feature-engineering)
    - [Model Selection & Training](https://github.com/tamaxhh/Customer-Churn-Prediction?tab=readme-ov-file#model-selection--training)
    - [Model Evaluation](https://github.com/tamaxhh/Customer-Churn-Prediction?tab=readme-ov-file#model-evaluation)
9. [Results & Discussion](https://github.com/tamaxhh/Customer-Churn-Prediction?tab=readme-ov-file#9-results--discussion)
10. [Conclusion & Future Work](https://github.com/tamaxhh/Customer-Churn-Prediction?tab=readme-ov-file#10-conclusion--future-work)
11. [Contributing](https://github.com/tamaxhh/Customer-Churn-Prediction?tab=readme-ov-file#11-contributing)
12. [License](https://github.com/tamaxhh/Customer-Churn-Prediction#12-license)

## 1. Project Overview

This project focuses on building a machine learning model to predict customer churn in a telecommunications company. By leveraging historical customer data, the aim is to identify at-risk customers proactively, enabling the implementation of targeted retention strategies to mitigate revenue loss and enhance customer satisfaction.

## 2. Problem Statement / Motivation

Customer churn poses a significant and persistent challenge for telecommunication companies. In a highly competitive market saturated with numerous service providers, customer acquisition costs are often substantially higher than customer retention costs. A high churn rate directly translates to:

- **Significant Revenue Loss:** Each churned customer represents lost recurring revenue, impacting the company's financial stability and growth projections.
- **Reduced Customer Lifetime Value (CLTV):** Churn diminishes the average CLTV, as customers are not retained long enough to realize their full revenue potential.
- **Increased Marketing & Acquisition Costs:** To offset churn, companies are forced to invest more heavily in acquiring new customers, which can be an inefficient use of resources if underlying retention issues are not addressed.
- **Brand Erosion & Negative Word-of-Mouth:** Dissatisfied or disengaged customers who churn may spread negative sentiments, damaging the company's reputation and deterring potential new customers.

Traditional reactive approaches to churn, such as exit surveys or last-ditch efforts, are often insufficient as they occur *after* the customer has already decided to leave. There is a critical need for a **proactive and data-driven approach** to identify customers at high risk of churning *before* they actually do.

This project aims to address this problem by leveraging historical customer data to develop a robust machine learning model capable of accurately predicting which customers are likely to churn. By identifying these "at-risk" customers early, the telecom company can implement targeted retention strategies, such as personalized offers, proactive customer support, or service adjustments, thereby mitigating revenue loss, enhancing customer satisfaction, and improving overall business profitability.

## 3. Dataset Description

The dataset utilized for this project contains 3,738 customer records from a telecom provider. It encompasses a comprehensive set of features categorized into:

- **Customer Demographics:** Information like `Gender`, `SeniorCitizen`, `Partner`, and `Dependents`.
- **Service Details:** Services the customer has subscribed to, such as `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, and `StreamingMovies`.
- **Account Information:** Details including `Tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, and `TotalCharges`.
- **Target Variable:** `Churn`, indicating whether the customer churned (`Yes`) or not (`No`).

**Key Dataset Statistics:**

- **Total Rows:** 3,738
- **Features:** 21
- **Target:** `Churn` (Binary: Yes/No)

## 4. Goals & Objectives

The primary goals of this project are to:

- **Explore Customer Behavior:** Conduct in-depth exploratory data analysis to understand customer behavior patterns and identify factors influencing churn.
- **Build a Predictive Model:** Develop and train a machine learning model capable of accurately predicting customer churn.
- **Identify Key Drivers:** Pinpoint the most influential features contributing to customer churn.
- **Suggest Actionable Recommendations:** Provide data-driven business recommendations for customer retention based on model insights.

## 5. Technologies Used

This project leverages the following technologies and libraries:

- **Python:** The primary programming language.
- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical operations.
- **Matplotlib:** For static, interactive, and animated visualizations.
- **Seaborn:** For statistical data visualization.
- **Scikit-learn:** For machine learning model building, preprocessing, and evaluation.

## 6. Installation & Setup

To run this Jupyter Notebook locally, follow these steps:

1. **Clone the repository:**
    
    ```
    git clone https://github.com/tamaxhh/Customer-Churn-Prediction.git
    cd customer-churn-prediction
    
    ```
    
2. **Create a virtual environment (recommended):**
    
    ```
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    
    ```
    
3. **Install the required libraries:**
    
    ```
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    
    ```
    
4. **Launch Jupyter Notebook:**
    
    ```
    jupyter notebook
    
    ```
    
5. **Open the notebook:** Navigate to `Customer Churn Prediction.ipynb` in your browser.

## 7. Project Structure

The repository contains the following key files:

- `Customer Churn Prediction.ipynb`: The main Jupyter Notebook containing all the code for data loading, preprocessing, EDA, model building, and evaluation.
- `README.md`: This file, providing an overview of the project.
- (Optional) `data/`: Directory for storing the dataset (if not directly included in the notebook).
- (Optional) `models/`: Directory for saving trained models.

## 8. Methodology

The project follows a standard machine learning pipeline:

### Data Loading & Initial Exploration

- The dataset is loaded using Pandas.
- Initial steps involve examining data types, checking for missing values, and understanding the basic structure of the dataset.

### Data Preprocessing

- **Missing Value Handling:** 5 rows with missing `TotalCharges` were dropped, assuming these represent incomplete customer records that would not significantly impact the overall analysis.
- **Categorical Encoding:** Categorical features (both nominal and ordinal) were converted into numerical representations suitable for machine learning models. This typically involves techniques like One-Hot Encoding for nominal features and Label Encoding/Manual Mapping for ordinal features.
- **Feature Scaling:** Numerical features were scaled to a standard range (e.g., using `StandardScaler`) to prevent features with larger magnitudes from disproportionately influencing model training.
- **Target Variable Preparation:** The `Churn` column was converted into a binary numerical format (e.g., 0 for 'No', 1 for 'Yes').

### Exploratory Data Analysis (EDA) Insights

Through extensive EDA, we uncovered critical patterns influencing customer churn:

- **Data Preprocessing:** Handled 5 missing values in `TotalCharges` by dropping rows. The `Churn` target variable is well-balanced.
- **Demographics:** Senior citizens, and customers without partners or dependents, show higher churn rates. Gender does not appear to be a significant factor.
- **Tenure & Contract:** New customers (low tenure) and those on month-to-month contracts exhibit significantly higher churn. Long-term contracts are strong retention factors.
- **Services & Charges:**
    - Customers lacking value-added services (e.g., Online Security, Tech Support) are more prone to churn.
    - Fiber optic internet users have a higher churn rate.
    - Higher `MonthlyCharges` and Electronic Check payment methods correlate with increased churn.

### Feature Engineering

*(Based on the provided snippets, there wasn't explicit feature engineering. If you added any, describe it here, e.g., "Created a new feature `ServiceCount` by summing binary service columns.")*

### Model Selection & Training

- A range of classification algorithms commonly used for churn prediction were considered, such as:
    - Logistic Regression
    - Decision Trees
    - Random Forest
    - Support Vector Machines (SVM)
    - Gradient Boosting Machines (e.g., LightGBM, XGBoost)
- The dataset was split into training and testing sets to ensure unbiased model evaluation.
- Models were trained on the preprocessed training data.
- Cross-validation techniques were employed to ensure model robustness and generalize well to unseen data.

### Model Evaluation

- Model performance was evaluated using appropriate metrics for classification problems, including:
    - **Accuracy:** Overall correctness of predictions.
    - **Precision:** Proportion of positive identifications that were actually correct.
    - **Recall (Sensitivity):** Proportion of actual positives that were identified correctly.
    - **F1-Score:** Harmonic mean of precision and recall.
    - **ROC AUC Score:** Area under the Receiver Operating Characteristic curve, indicating the model's ability to distinguish between classes.
- Confusion matrices and ROC curves were used to visualize model performance.

## 9. Results & Discussion

- **Model Performance:** The trained model achieved a strong performance, demonstrating its capability to accurately classify customers as churners or non-churners. *(Detail specific metrics here, e.g., "The Random Forest model yielded an accuracy of X%, a precision of Y%, and a recall of Z% on the test set. The AUC score was 0.XX, indicating strong discriminatory power.")*
- **Feature Importance:** Analysis of feature importance (e.g., from tree-based models) revealed the most influential factors driving churn. Consistent with EDA, `Contract` type, `TechSupport`, `OnlineSecurity`, `Tenure`, and `MonthlyCharges` emerged as top predictors.
- **Business Impact:** The insights gained from the model can directly inform business strategies. For example, identifying the critical role of specific services (Tech Support, Online Security) suggests that promoting these or including them in bundles could significantly improve retention. The high churn among new customers highlights the need for robust onboarding and early engagement programs.

## 10. Conclusion & Future Work

This project successfully developed a machine learning model to predict customer churn for a telecom company, revealing key actionable insights:

- **Model Performance:** The model effectively predicts customer churn, providing a valuable tool for proactive retention.
- **Primary Churn Drivers:**
    - **Contract Type:** Month-to-month contracts are the strongest predictor of churn.
    - **Value-Added Services:** Absence of services like Tech Support and Online Security significantly increases churn risk.
    - **Tenure:** Newer customers have a higher propensity to churn.
    - **Internet Service & Charges:** Fiber optic users and those with higher monthly charges show increased churn.
- **Business Recommendations:**
    - **Incentivize Long-Term Contracts:** Encourage transitions from month-to-month plans.
    - **Promote Security & Support Services:** Highlight the value of retention-boosting services.
    - **Target New Customers:** Implement specific onboarding and early retention programs.
    - **Analyze Service Specifics:** Investigate churn drivers within the Fiber Optic segment and electronic payment methods.
- **Future Enhancements:** Explore advanced feature engineering, evaluate more complex models (e.g., XGBoost), and integrate real-time prediction for immediate intervention.

## 11. Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to open an issue or submit a pull request.

## 12. License

This project is licensed under the MIT License - see the `LICENSE` file for details.
