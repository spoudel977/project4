#CREDIT RISK CLASSIFICATION with MACHINE LEARNING MODELS

![360_F_258871009_f5net6t178mMF1nekdg2AS2vuOUhpDjL](https://github.com/spoudel977/project4/assets/105176210/a2e7f5cc-ee19-417c-8d97-a09e9b321da0)

# Overview of the Project 
   1. Purpose
      
      Our goal with this dataset is to create a machine learning model which helps to indicate which applicants would be eligible for a loan according to their credit score. The credit score will based on the different data that has been provided for each applicant. Our model will determine the applicant's credit worthiness for a loan based on applicant's financial overview (monthly income, previous loans and outstanding payments, age, occupation, etc.).
   
   3. Datasets used 
        
      Dataset: https://www.kaggle.com/datasets/parisrohan/credit-score-classification
   
   
   4. Data cleaning 
    - Dropped unnecessary columns
    - Checked for missing values: df = df.insull().sum()
    - Dropped Missing Values: df.dropna()
    - Converted Dataypes into int, float, obj
    - Converted Categorical Data into labelled number: Good (0), Poor (1), and Standard (2)

    
    
   5. Training and Evaluating Models 
   
   

# Overview of the Model Analysis 
    *Summary of all model accuracies: *
    Logistic Regression: 64.71%
    Decision Tree: 71.41%
    Random Forest: 72.68%
    Gradient Boosting: 71.38%
    Support Vector Machine: 70.45%
    Neural Network: 70.25%
    
    
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
        
