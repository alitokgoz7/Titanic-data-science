# Titanic Survival Prediction

## Project Overview

This project analyzes the Titanic passenger dataset and builds a machine learning model to predict passenger survival.

The project covers the complete introductory data science workflow, including data exploration, exploratory data analysis (EDA), data cleaning, feature engineering, data preparation, and machine learning with Logistic Regression.

## Dataset

The project uses the Titanic training dataset containing **891 passengers and 12 original features**.

The target variable is:

* `Survived` — indicates whether a passenger survived (`1`) or did not survive (`0`).

Some of the main passenger information includes:

* Passenger class (`Pclass`)
* Sex
* Age
* Number of siblings/spouses (`SibSp`)
* Number of parents/children (`Parch`)
* Fare
* Cabin information
* Embarkation port (`Embarked`)

## Exploratory Data Analysis

Exploratory data analysis was performed to better understand the dataset and investigate factors associated with passenger survival.

The analysis included:

* Overall survival distribution
* Survival rates by sex
* Survival rates by passenger class
* Age distribution
* Fare distribution
* Survival rates by embarkation port
* Survival patterns across sex and passenger class
* Family size and solo travel analysis

## Data Cleaning

The dataset contained missing values mainly in `Age`, `Cabin`, and `Embarked`.

Different strategies were used depending on the characteristics of each feature:

* Missing `Age` values were filled using median ages based on passenger titles.
* Missing `Embarked` values were filled using the most frequent embarkation port.
* `Cabin` was not directly imputed because of its large number of missing values. Instead, cabin availability was converted into a new feature.

## Feature Engineering

Several new features were created to extract additional information from the original dataset.

### Title

Passenger titles such as `Mr`, `Mrs`, `Miss`, and `Master` were extracted from the `Name` column.

### HasCabin

A binary feature was created to indicate whether cabin information was available.

### FamilySize

Total family size was calculated using:

`SibSp + Parch + 1`

### IsAlone

A binary feature was created to identify passengers traveling alone.

### FareperPerson

The fare paid per person was estimated using the passenger's fare and family size.

## Machine Learning Preparation

Before training the model:

* `Survived` was separated as the target variable.
* Columns such as `PassengerId`, `Name`, `Ticket`, and `Cabin` were excluded from the initial model.
* Categorical variables were converted into numerical representations.
* The dataset was divided into training and validation sets.

## Model

A **Logistic Regression** model was used as the first classification model.

```python
model = LogisticRegression(max_iter=1000)
```

The model was trained using the training dataset and evaluated on the validation dataset.

Predicted survival values were compared with the actual validation values, and model performance was evaluated using accuracy.

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Project Structure

```text
Titanic-data-science/
│
├── data/
│   └── train.csv
│
├── notebooks/
│   └── Titanic_analysis.ipynb
│
├── .gitignore
└── README.md
```

## Future Improvements

This project currently uses Logistic Regression as the first baseline machine learning model.

Possible next steps include:

* Confusion matrix analysis
* Precision, recall, and F1-score evaluation
* Feature scaling where appropriate
* Testing additional classification algorithms
* Comparing model performances
* Hyperparameter tuning
* Cross-validation
* Improving feature engineering

## Author

Developed as a data science and machine learning project using the Titanic dataset.
