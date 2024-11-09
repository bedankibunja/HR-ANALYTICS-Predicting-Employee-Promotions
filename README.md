# HR-ANALYTICS-Predicting-Employee-Promotions

![Employee Prediction](./Images/Employee_Prediction.png "Employees sitting around a table")

## Project Overview
This project applies predictive analytics to streamline the promotion process in a large multinational corporation (MNC) with multiple organizational verticals. Currently, identifying candidates for promotion relies heavily on manual HR evaluations and KPI assessments, which are time-consuming and delay decision-making. By building a predictive model, this project aims to help the HR department proactively identify employees eligible for promotion, accelerating their career progression and enhancing HR efficiency.

## Business Understanding
Our client, a large MNC, operates across 9 broad organizational verticals. A primary challenge they face is identifying high-potential candidates for promotion in a timely manner. Currently, employees are recommended for promotion based on past performance and are required to complete training and evaluation programs specific to each vertical. Employees who achieve at least 60% KPI completion are considered for promotion, but final decisions are only made after all evaluations are completed, delaying the promotion cycle.

The goal is to help HR departments proactively identify employees eligible for promotion at a specific checkpoint, enabling a faster, data-driven promotion cycle.

## Problem Statement
The promotion process for managerial roles and below is manual and labor-intensive, resulting in delays that affect both employee morale and organizational agility. This project develops a predictive model using demographic and performance data to assist HR in making more efficient, data-driven promotion decisions.

## Objectives

### Main Objective
- **To improve the accuracy and efficiency of the promotion process** by building a machine learning model to identify employees most likely to be promoted.

### Key Business Questions
1. Which employees are most likely to be promoted in the next cycle?
2. What are the primary factors influencing promotion decisions?
3. Are there biases in promotion practices that need addressing?
4. How can training programs be adjusted to boost promotion readiness?
5. How does the likelihood of promotion affect employee retention?

## Stakeholders
- **HR Department**: Primary users of the promotion prediction model.
- **Department Heads and Team Managers**: To understand promotion potential within their teams.
- **Executive Leadership**: For strategic planning and talent management.
- **Data Science/IT Team**: Responsible for maintaining and integrating the model within HR systems.

## Data Understanding
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

## Methodology
The project includes the following steps:
1. **Exploratory Data Analysis (EDA)**: Understanding data distributions and identifying patterns.
2. **Feature Engineering**: Creating new features to enhance model performance.
3. **Data Preprocessing**: Handling missing values, encoding categorical features, and scaling data.
4. **Model Selection and Training**: Testing multiple algorithms (e.g., Logistic Regression, Decision Trees) to find the best-performing model.
5. **Evaluation**: Assessing model performance based on F1 Score, Precision, and Recall.

## Results and Findings
The README will be updated with specific results and findings upon model completion, including key metrics and any actionable insights for HR departments.

## Conclusion
This project provides a framework for MNCs to streamline their promotion process using data-driven insights, improving HR operations and facilitating employee career growth.

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