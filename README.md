# HR-ANALYTICS-Predicting-Employee-Promotions

![Employee Prediction](./Images/Employee_Prediction.png "Employees sitting around a table")


## Table of Contents
- [Project Overview](#project-overview)
- [Business Understanding](#business-understanding)
  - [Business Problem](#business-problem)
  - [Stakeholders](#stakeholders)
- [Data Description](#data-description)
- [Objectives](#objectives)
  - [Main Objectives](#main-objectives)
  - [Key Business Questions](#key-business-questions)
- [Methodology](#methodology)
- [Results and Findings](#results-and-findings)
- [Conclusion](#conclusion)
  - [Findings](#findings)
  - [Recommendations](#recommendations)
- [Future Work](#future-work)
- [How to Use This Repository](#how-to-use-this-repository)
- [Requirements](#requirements)
- [Contributors](#contributors)

## Project Overview
This project applies predictive analytics to streamline the promotion process in a large multinational corporation (MNC) with multiple organizational verticals. Currently, identifying candidates for promotion relies heavily on manual HR evaluations and KPI assessments, which are time-consuming and delay decision-making. By building a predictive model, this project aims to help the HR department proactively identify employees eligible for promotion, accelerating their career progression and enhancing HR efficiency.

## Business Understanding

### Business Problem
The promotion process for managerial roles and below is manual and labor-intensive, resulting in delays that affect both employee morale and organizational agility. This project develops a predictive model using demographic and performance data to assist HR in making more efficient, data-driven promotion decisions.

### Stakeholders
- **HR Department**: Primary users of the promotion prediction model.
- **Department Heads and Team Managers**: To understand promotion potential within their teams.
- **Executive Leadership**: For strategic planning and talent management.
- **Data Science/IT Team**: Responsible for maintaining and integrating the model within HR systems.

## Data Description
The project uses two datasets:

1. **Train Dataset**: Contains information on current employees and a target variable indicating if they were promoted.
2. **Test Dataset**: Similar data without the target label, used for model evaluation.

### Key Features
- **employee_id**: Unique identifier.
- **department**: Employee's department.
- **region**: Location of the employee’s role.
- **education**: Highest level of education.
- **gender**: Gender of the employee.
- **recruitment_channel**: Method of recruitment.
- **no_of_trainings**: Number of trainings completed in the last year.
- **age**: Employee's age.
- **previous_year_rating**: Performance rating for the prior year.
- **length_of_service**: Tenure in the organization.
- **KPIs_met >80%**: Indicator of KPI completion over 80%.
- **awards_won?**: Indicator of any awards received in the past year.
- **avg_training_score**: Average score in recent training evaluations.
- **is_promoted**: Target variable indicating promotion recommendation (1 for promoted, 0 otherwise).

## Objectives

### Main Objectives
- **To improve the accuracy and efficiency of the promotion process** by building a machine learning model to identify employees most likely to be promoted.

### Key Business Questions
1. Which employees are most likely to be promoted in the next cycle?
2. What are the primary factors influencing promotion decisions?
3. Are there biases in promotion practices that need addressing?
4. How can training programs be adjusted to boost promotion readiness?
5. How does the likelihood of promotion affect employee retention?

## Methodology

1. **Exploratory Data Analysis (EDA)**: 
In the notebook, EDA uses visualizations and correlation analyses to explore distributions and relationships among features, identifying trends that influence promotion, such as `previous_year_rating` and `avg_training_score`. This analysis informs feature engineering and data cleaning steps.

2. **Feature Engineering**: 
New features are created based on identified patterns, such as combining training-related variables and regrouping sparse categories in `department`. These engineered features aim to improve the model’s ability to predict promotion potential effectively.

3. **Data Preprocessing**: 
The notebook preprocesses data by imputing missing values, encoding categorical features, and scaling numerical features. These steps create a structured dataset ready for model training, ensuring consistency and compatibility for machine learning.

4. **Model Selection and Training**: 
Multiple models, including Logistic Regression and Random Forest, are trained with tuned hyperparameters. Cross-validation is used to identify the best-performing model based on predictive accuracy and generalizability.

5. **Evaluation**: 
The model’s performance is evaluated using F1 Score, Precision, and Recall, with a focus on balancing prediction accuracy and reducing false negatives. A confusion matrix provides additional insight into model errors, guiding future improvements.

## Results and Findings
The README will be updated with specific results and findings upon model completion, including key metrics and any actionable insights for HR departments.

## Conclusion

### Findings
This section will be populated with insights gained from the analysis.

### Recommendations
This section will include actionable recommendations based on model results.

## Future Work
Future improvements may include fine-tuning the model, implementing bias mitigation strategies, and exploring more complex algorithms.

## How to Use This Repository
1. Clone the repository.
2. Run the notebook file `Notebook.ipynb` in a Jupyter Notebook environment.
3. Install necessary packages as outlined in the requirements section below.

## Requirements
- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

To install the requirements, run:
```bash
pip install -r requirements.txt