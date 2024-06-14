# CREDIT RISK CLASSIFICATION with MACHINE LEARNING MODELS

![360_F_258871009_f5net6t178mMF1nekdg2AS2vuOUhpDjL](https://github.com/spoudel977/project4/assets/105176210/a2e7f5cc-ee19-417c-8d97-a09e9b321da0)

# Overview of the Project 
1. Purpose   

The goal of the project is to create a machine learning model which helps to indicate which applicants would be eligible for a loan according to their credit score. The credit score is tested against 13 features that has been provided for each applicant. Our model will determine the applicant's credit worthiness for a loan based on the applicant's financial overview ( features): Age, Annual_Income, Monthly_Inhand_Salary, Num_Bank_Accounts, Num_Credit_Card, Interest_Rate, Num_of_Loan, Delay_from_due_date, Num_of_Delayed_Payment, Changed_Credit_Limit, Credit_Mix,.. The project is implemented in a Jupyter Notebook, using the Python programming language and several popular machine learning libraries such as NumPy, Pandas, Matplotlib, Seaborn, sklearn.preprocessing, sklearn.tree and sklearn.ensemble. 

2. Datasets used
The data used in this project is a simulated dataset of individuals and their credit scores.
 [Dataset: Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classificationD) - train.csv

      
3. Libraries Used 
      - pandas
      - NumPy
      - Matplotlib
      - Seaborn
      - Scikit-learn
      - Imbalanced-learn
      - TensorFlow

                     
4. Data Exploration, Cleaning and Visualisations
    - Dropped unnecessary columns
    - Checked for missing values: df = df.insull().sum()
    - Dropped Missing Values: df.dropna()
    - Converted Dataypes into int, float, obj
    - Converted Categorical Data into labelled number: Good (0), Poor (1), and Standard (2) using Encoding
    - Rounding Float columns upto 2 decimals.
    - Balanced Data using oversampling
    - Dropped unwanted outliers 
    - Plotted boxplots to check skewness of data
    - Standarized and Normalised data


6. Model explorations and selection
   - Neural Network
   - Random Forest
   - Logistic Regression
   - Decision Tree
   - Gradient Boosting
 
6.1 Neural Network
    Achieved accuracy of 75%. When used Batch Normalization for optimization accuracy rate decreased to 72.25%. Further optimization not progressed. Further optimisation is suggested.

6.2 Random Forest

In validation Set results, achieved overall accuracy of 91%. Class 0 (Poor) had high precision rate of 99.0% confirming model prediction is almost always correct. Very high recall of 99.9% suggesting the model successfully identifies nearly all instances of 'Poor Credit Scores".
CLass 1 (Standard) had precision rate of 91% indicating most of the predictions for standards were correct. High recall of 97%  tells model correctly indentifying majority of 'Standard' instances. 
Class 2 (Good) had precision rate of 98.1%  indicating model accurracy as high however recall of 75%  tells missing a number of 'Good' Credit Score instances. While plotted confusion matrix, a higher number of data incorrectly classified in 'poor' and 'standard'. 

![ConfusionMatrix1](https://github.com/spoudel977/project4/assets/94360634/4127951b-ec8c-4efb-a2f6-ace57a2fdf7e)

Synthetic Minority Over-sampling Technique (SMOT) applied to solve problem of imbalanced dateset by generating synthetic samples for the minority class. Model retrained and tested. Identified SMOTE technique did not significantly alter the model's performance as there were minimal changes (slight improvement in recall for Class 2) in accuracy, precision, recall and F1 score.

To summarise the model is consistenly performing well across all metrics both before and after the optimization indicating that the initial model was already well-tuned because of rigorous data exploration and cleaning and SMOT provided a minor enhancement in balancing the class predictions.

Feature importances was visualized as below and identified topfluences: Outstanding Debt, Interest Rate, Payment of Minimum Amount, Credit Mix. Lower influencers also noted as Payment Behaviour in different categories. Team understands further optimization requires to address issue of incorrect classification of good score.
![output](https://github.com/spoudel977/project4/assets/94360634/53ca331e-9a47-4bcf-b485-805aee328402)


    Summary of all model accuracies:
    Logistic Regression: 64.71%
    Decision Tree: 71.41%
    Random Forest: 72.68%
    Gradient Boosting: 71.38%
    Support Vector Machine: 70.45%
    Neural Network: 72.25%
    
    
# Optimization of the Model 
       Best parameters found:  {'bootstrap': True, 'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}
        Accuracy: 76.56
        Classification Report:
                      precision    recall  f1-score   support

                   0       0.63      0.66      0.64      1069
                   1       0.78      0.76      0.77      1960
                   2       0.80      0.80      0.80      3661

            accuracy                           0.77      6690
           macro avg       0.74      0.74      0.74      6690
        weighted avg       0.77      0.77      0.77      6690

        }
        
