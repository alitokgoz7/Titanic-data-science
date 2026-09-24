# 🚢 Titanic Survival Prediction

An end-to-end machine learning project that predicts passenger survival on the Titanic dataset.

The project covers the complete workflow from **exploratory data analysis and feature engineering to model training, evaluation, prediction, and a simple web-based frontend for presenting the project results.**

---

## 📌 Project Overview

The main objective of this project is to predict whether a Titanic passenger survived based on passenger information such as age, gender, passenger class, fare, family information, and cabin availability.

The project includes:

* Exploratory Data Analysis (EDA)
* Data preprocessing
* Feature engineering
* Logistic Regression
* Decision Tree
* Random Forest
* Cross-validation
* Hyperparameter tuning
* Model evaluation
* Prediction pipeline
* HTML/CSS/JavaScript frontend

---

## 🔎 Exploratory Data Analysis

The Titanic training dataset contains **891 passengers**.

During EDA, the dataset was analyzed for:

* Missing values
* Feature distributions
* Survival rates
* Numerical and categorical variables
* Relationships between passenger characteristics and survival

Important missing values were found in `Age`, `Cabin`, and `Embarked`.

---

## 🛠 Feature Engineering

New features were created from the original passenger information to improve the dataset used by the models.

### `Title`

Passenger titles were extracted from the `Name` column and grouped into:

```text
Mr
Miss
Mrs
Master
Rare
```

### `FamilySize`

```text
FamilySize = SibSp + Parch + 1
```

Represents the total number of family members travelling together.

### `IsAlone`

Indicates whether the passenger was travelling alone.

### `HasCabin`

Indicates whether cabin information exists for the passenger.

### `FareperPerson`

```text
FareperPerson = Fare / FamilySize
```

Represents the approximate fare paid per family member.

---

## 🧹 Data Preprocessing

The preprocessing pipeline prepares the data for machine learning.

Main preprocessing steps:

* Missing `Embarked` values filled using the mode
* Missing `Age` values filled using median age by passenger title
* `Sex` converted into numerical values
* `Embarked` one-hot encoded
* `Title` one-hot encoded
* Unnecessary columns removed

The final model uses **18 features**.

---

## 🤖 Machine Learning Models

Three classification algorithms were compared:

| Model               | Training Accuracy | Validation Accuracy |
| ------------------- | ----------------: | ------------------: |
| Logistic Regression |            83.85% |          **84.36%** |
| Decision Tree       |            98.74% |              78.77% |
| Random Forest       |            98.74% |              78.77% |
| Tuned Random Forest |            92.56% |              81.01% |

The Decision Tree and Random Forest models showed signs of overfitting.

Random Forest was also optimized using `GridSearchCV`.

Based on validation performance and generalization, **Logistic Regression was selected as the final model.**

---

## 📊 Final Model Performance

Logistic Regression achieved approximately:

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 0.844 |
| Precision | 0.806 |
| Recall    | 0.783 |
| F1 Score  | 0.794 |
| ROC-AUC   | 0.874 |

5-fold cross-validation produced an average accuracy of approximately **82.6%**.

The trained model is saved using `joblib`:

```text
outputs/logistic_regression_model.pkl
```

---

## 🔮 Prediction Pipeline

The project includes a separate prediction pipeline for unseen passenger data.

```text
Passenger Data
      ↓
Feature Engineering
      ↓
Data Preprocessing
      ↓
Feature Alignment
      ↓
Logistic Regression Model
      ↓
Survival Prediction
```

The same preprocessing and feature engineering logic used during training is applied to the test data before predictions are generated.

---

## 🌐 Web Interface — `index.html`

The project also includes an `index.html` page that provides a simple **frontend for presenting the machine learning project and its results**.

While Python handles the data processing, model training, evaluation, and prediction workflow, the web interface provides a more visual and user-friendly way to explore the project.

The frontend uses:

* **HTML** — page structure and content
* **CSS** — layout and visual design
* **JavaScript** — interactive page behavior

The frontend acts as the **presentation layer** of the project.

```text
Machine Learning
Python / Scikit-learn
        ↓
Model Results
        ↓
HTML / CSS / JavaScript
        ↓
Web Interface
```

The page can be opened directly in a browser using:

```text
index.html
```

> The current frontend is used to present the project and its results. The Scikit-learn model itself runs in Python and is not executed directly by the browser.

---

## 📁 Project Structure

```text
Titanic-data-science/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── Exploratory analysis and model experiments
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── outputs/
│   └── logistic_regression_model.pkl
│
├── index.html
├── Readme.md
└── project_context.md
```

### Source Files

* **`features.py`** — creates engineered features such as `Title`, `FamilySize`, `IsAlone`, `HasCabin`, and `FareperPerson`.
* **`preprocessing.py`** — handles missing values, categorical encoding, and data preparation.
* **`train.py`** — trains and saves the Logistic Regression model.
* **`evaluate.py`** — calculates model evaluation metrics.
* **`predict.py`** — loads the trained model and generates predictions for new data.

---

## 🛠 Technologies

**Data Science & Machine Learning**

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn` · `Joblib`

**Frontend**

`HTML` · `CSS` · `JavaScript`

**Development**

`Jupyter Notebook` · `Git` · `GitHub`

---

## ▶️ Running the Project

Clone the repository:

```bash
git clone https://github.com/alitokgoz7/Titanic-data-science.git
cd Titanic-data-science
```

Train the model:

```bash
python src/train.py
```

The trained model will be saved inside the `outputs` directory.

To view the frontend, open:

```text
index.html
```

in a web browser.
