**# Titanic Survival Prediction**

**## Project Overview**

This project implements an end-to-end machine learning workflow for predicting passenger survival on the Titanic.

The project covers the complete data science lifecycle, including exploratory data analysis (EDA), data cleaning, feature engineering, preprocessing, model training, model comparison, cross-validation, hyperparameter tuning, model evaluation, and final prediction generation.

The initial analysis and experimentation were performed in a Jupyter Notebook. After the modeling stage, the workflow was refactored into reusable Python modules for preprocessing, feature engineering, training, evaluation, and prediction.

\---

**## Dataset**

The project uses the Titanic dataset consisting of:

\* \`train.csv\` — 891 passengers with survival labels

\* \`test.csv\` — 418 passengers used for final predictions

The target variable is:

\* \`Survived\`

  \* \`0\` — Did not survive

  \* \`1\` — Survived

Important original features include:

\* \`Pclass\` — Passenger class

\* \`Sex\` — Passenger sex

\* \`Age\` — Passenger age

\* \`SibSp\` — Number of siblings/spouses aboard

\* \`Parch\` — Number of parents/children aboard

\* \`Fare\` — Passenger fare

\* \`Cabin\` — Cabin information

\* \`Embarked\` — Port of embarkation

\---

**## Exploratory Data Analysis**

Exploratory data analysis was performed to understand the dataset, identify missing values, examine distributions, and investigate relationships between passenger characteristics and survival.

The analysis included:

\* Overall survival distribution

\* Survival rate by sex

\* Survival rate by passenger class

\* Age distribution

\* Fare distribution

\* Survival rate by embarkation port

\* Survival patterns across sex and passenger class

\* Family size analysis

\* Solo traveler analysis

\* Missing value analysis

\* Feature relationships with survival

\---

**## Data Cleaning**

Missing values were handled according to the characteristics of each feature.

**### Age**

Passenger titles were extracted from names and missing age values were filled using the median age of passengers with the same title.

The main title groups were:

\* \`Mr\`

\* \`Mrs\`

\* \`Miss\`

\* \`Master\`

\* \`Rare\`

**### Embarked**

Missing \`Embarked\` values were filled using the most frequent embarkation port.

**### Fare**

Missing fare values were filled using the median fare.

**### Cabin**

Because a large proportion of the \`Cabin\` column was missing, the column was not directly imputed. Instead, cabin availability was converted into a binary feature called \`HasCabin\`.

\---

**## Feature Engineering**

Several new features were created from the original dataset.

**### Title**

Passenger titles were extracted from the \`Name\` column.

Less common titles were grouped into a single \`Rare\` category.

**### FamilySize**

Total family size was calculated as:

\`\`\`text

FamilySize = SibSp + Parch + 1

\`\`\`

**### IsAlone**

A binary feature indicating whether the passenger was traveling alone.

\`\`\`text

1 = Traveling alone

0 = Traveling with family

\`\`\`

**### HasCabin**

A binary feature indicating whether cabin information was available.

\`\`\`text

1 = Cabin information available

0 = Cabin information missing

\`\`\`

**### FareperPerson**

The fare paid per person was calculated using:

\`\`\`text

FareperPerson = Fare / FamilySize

\`\`\`

\---

**## Data Preprocessing**

Before model training:

\* Missing values were handled

\* Passenger titles were extracted and grouped

\* New features were generated

\* \`Sex\` was converted into numerical values

\* \`Embarked\` was one-hot encoded

\* \`Title\` was one-hot encoded

\* Unnecessary columns were removed

\* The target variable was separated from the features

\* The dataset was split into training and validation sets

The train-validation split used:

\`\`\`python

train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

\`\`\`

\---

**## Machine Learning Models**

Three classification algorithms were evaluated.

**### Logistic Regression**

Logistic Regression was used as the primary baseline classification model.

**### Decision Tree**

A Decision Tree classifier was tested to compare a tree-based model with Logistic Regression.

The model achieved very high training accuracy but lower validation accuracy, indicating overfitting.

**### Random Forest**

Random Forest was evaluated as an ensemble-based alternative.

Hyperparameter tuning was also performed using GridSearchCV.

The best tested parameters were:

\`\`\`text

n_estimators = 100

max_depth = 10

min_samples_split = 5

\`\`\`

\---

**## Model Comparison**

\| Model               | Training Accuracy | Validation Accuracy |

\| ------------------- | ----------------: | ------------------: |

\| Logistic Regression |            0.8385 |              0.8436 |

\| Decision Tree       |            0.9874 |              0.7877 |

\| Random Forest       |            0.9874 |              0.7877 |

\| Tuned Random Forest |            0.9256 |              0.8101 |

Logistic Regression showed the strongest validation performance among the tested models while maintaining similar training and validation accuracy.

Decision Tree and the initial Random Forest model showed signs of overfitting due to the large difference between training and validation performance.

\---

**## Final Model Evaluation**

Logistic Regression was selected as the final model.

Validation results:

\| Metric    |  Score |

\| --------- | -----: |

\| Accuracy  | 0.8436 |

\| Precision | 0.8060 |

\| Recall    | 0.7826 |

\| F1 Score  | 0.7941 |

\| ROC-AUC   | 0.8736 |

**### Confusion Matrix**

\`\`\`text

[[97 13]

 [15 54]]

\`\`\`

This corresponds to:

\* True Negatives: 97

\* False Positives: 13

\* False Negatives: 15

\* True Positives: 54

\---

**## Cross-Validation**

Five-fold cross-validation was used to evaluate the stability of the Logistic Regression model.

Cross-validation accuracy scores:

\`\`\`text

0.8268

0.8090

0.7978

0.8202

0.8764

\`\`\`

Mean cross-validation accuracy:

\`\`\`text

0.8260

\`\`\`

Standard deviation:

\`\`\`text

0.0271

\`\`\`

The cross-validation results indicate relatively consistent performance across different subsets of the training data.

\---

**## Feature Analysis**

Logistic Regression coefficients were analyzed to understand how different features contributed to model predictions.

Some of the strongest positive coefficients included:

\* \`Title_Master\`

\* \`HasCabin\`

\* \`Title_Mrs\`

Some of the strongest negative coefficients included:

\* \`Title_Mr\`

\* \`Sex\`

\* \`Pclass\`

\* \`Title_Rare\`

\* \`IsAlone\`

These coefficients describe relationships learned by the model and should not be interpreted as causal effects.

\---

**## Final Prediction Pipeline**

After model selection, the workflow was refactored from notebook-based experimentation into reusable Python modules.

The final pipeline performs:

\`\`\`text

Raw Data

   ↓

Feature Engineering

   ↓

Missing Value Handling

   ↓

Categorical Encoding

   ↓

Model Training

   ↓

Model Evaluation

   ↓

Model Serialization

   ↓

Test Data Preprocessing

   ↓

Prediction

   ↓

submission.csv

\`\`\`

The trained Logistic Regression model is serialized using \`joblib\`.

The prediction script loads the saved model, preprocesses \`test.csv\`, generates survival predictions, and creates:

\`\`\`text

outputs/submission.csv

\`\`\`

\---

**## Project Structure**

\`\`\`text

Titanic-data-science/

│

├── data/

│   ├── train.csv

│   └── test.csv

│

├── notebooks/

│   └── Titanic_analysis.ipynb

│

├── outputs/

│   ├── logistic_regression_model.pkl

│   └── submission.csv

│

├── src/

│   ├── preprocessing.py

│   ├── features.py

│   ├── train.py

│   ├── evaluate.py

│   └── predict.py

│

├── index.html

├── requirements.txt

├── .gitignore

└── Readme.md

\`\`\`

**### Module Responsibilities**

\`features.py\`

Creates engineered features such as \`Title\`, \`FamilySize\`, \`IsAlone\`, \`HasCabin\`, and \`FareperPerson\`.

\`preprocessing.py\`

Handles data loading, missing values, categorical encoding, feature preparation, and removal of unnecessary columns.

\`train.py\`

Loads and prepares the training data, splits the dataset, trains the Logistic Regression model, evaluates its performance, and saves the trained model.

\`evaluate.py\`

Calculates classification metrics including accuracy, precision, recall, F1 score, confusion matrix, and ROC-AUC.

\`predict.py\`

Loads the trained model, preprocesses the Titanic test dataset, generates predictions, and creates the final submission file.

\---

**## Installation**

Clone the repository and move into the project directory:

\`\`\`bash

git clone \<repository-url>

cd Titanic-data-science

\`\`\`

Create and activate a virtual environment:

\`\`\`bash

python -m venv .venv

source .venv/bin/activate

\`\`\`

Install the required dependencies:

\`\`\`bash

pip install -r requirements.txt

\`\`\`

\---

**## Requirements**

The main dependencies are:

\`\`\`text

pandas==3.0.5

scikit-learn==1.9.0

joblib==1.6.0

\`\`\`

Additional libraries such as NumPy, Matplotlib, Seaborn, and Jupyter were used during exploratory data analysis and notebook-based experimentation.

\---

**## Usage**

**### Train and Evaluate the Model**

From the project root directory:

\`\`\`bash

python src/train.py

\`\`\`

This trains the Logistic Regression model, prints the validation metrics, and saves the trained model to:

\`\`\`text

outputs/logistic_regression_model.pkl

\`\`\`

**### Generate Predictions**

Run:

\`\`\`bash

python src/predict.py

\`\`\`

The script loads the trained model and generates:

\`\`\`text

outputs/submission.csv

\`\`\`

\---

**## Technologies**

\* Python

\* Pandas

\* NumPy

\* Scikit-learn

\* Matplotlib

\* Seaborn

\* Jupyter Notebook

\* Git

\* GitHub

\---

**## Key Learning Outcomes**

This project demonstrates practical experience with:

\* Exploratory Data Analysis

\* Data Cleaning

\* Missing Value Handling

\* Feature Engineering

\* Categorical Encoding

\* Binary Classification

\* Logistic Regression

\* Decision Trees

\* Random Forests

\* Cross-Validation

\* Hyperparameter Tuning

\* Classification Metrics

\* ROC-AUC Evaluation

\* Overfitting Analysis

\* Model Serialization

\* Modular Python Project Structure

\* End-to-End Machine Learning Workflows

\---

**## Future Improvements**

Possible extensions include:

\* Building preprocessing and modeling with Scikit-learn \`Pipeline\`

\* Using \`ColumnTransformer\` for more robust preprocessing

\* Preventing preprocessing information from being learned outside individual cross-validation folds

\* Testing additional boosting algorithms

\* Performing more extensive hyperparameter optimization

\* Adding automated tests for preprocessing and feature engineering

\* Adding experiment tracking

\* Containerizing the project for reproducible deployment

\---

**## Author**

Developed as an end-to-end data science and machine learning project using the Titanic dataset.