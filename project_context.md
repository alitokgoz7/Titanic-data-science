# Titanic Data Science Project — Context

## Project Goal

This is a step-by-step Data Science learning project using the Titanic dataset.

The main goal is to learn the complete workflow:

Data Understanding → EDA → Data Cleaning → Feature Engineering → Machine Learning → Model Evaluation

The project should gradually become a professional GitHub portfolio project.

## Learning Style

I want to write the code myself.

When giving me a task:

1. Explain what I need to achieve.
2. Give a small hint if necessary.
3. Do NOT immediately provide the complete solution.
4. Let me attempt the code first.
5. If my code is wrong, explain why and help me correct it.
6. Explain new pandas / Python / ML concepts when they first appear.
7. Suggest short, professional English comments I can add above important notebook cells.

We progress in batches of approximately 10 tasks.

---

## Project Structure

Repository:

`Titanic-data-science`

Main dataset:

`train.csv`

The analysis is currently being performed in a Jupyter Notebook.

---

# Current Progress

## Tasks 1–10 — Data Understanding

Completed.

Topics included:

* Loading the dataset
* Inspecting the DataFrame
* Dataset shape
* Column names
* Data types
* Basic descriptive statistics
* Unique values
* Number of unique values
* Understanding categorical and numerical variables

---

# Tasks 11–20 — Exploratory Data Analysis

## Task 11 — Missing Values ✅

Used:

`df.isnull().sum()`

Results:

* Age: 177 missing
* Cabin: 687 missing
* Embarked: 2 missing

Cabin has the largest number of missing values.

---

## Task 12 — Missing Value Percentage ✅

Calculated missing-value percentages using the number of rows in the dataset.

Dataset contains:

891 rows

Concept learned:

`len(df)` returns the number of rows in the DataFrame.

---

## Task 13 — Duplicate Rows ✅

Used:

`df.duplicated().sum()`

Result:

0 duplicate rows.

---

## Task 14 — Survival Distribution ✅

Analyzed the `Survived` column using `value_counts()`.

Results:

* 0 (did not survive): 549
* 1 (survived): 342

Also calculated percentages using:

`normalize=True`

Approximately 38.4% of passengers survived.

---

# Tasks 15–20 — Exploratory Data Analysis

Completed.

Topics included:

* Survival rates by sex
* Survival rates by passenger class
* Age summary and distribution
* Fare summary, skewness, and outliers
* Survival rates by embarkation port
* Survival rates by sex and passenger class

---

# Tasks 21–25 — Feature Engineering and Data Cleaning

## Task 21 — Extract Title from Name ✅

Create a new `Title` feature from the `Name` column.

## Task 22 — Analyze Titles ✅

Examine the distribution of titles and median age for each title.

## Task 23 — Fill Missing Age Values ✅

Fill missing `Age` values with the median age of the passenger's `Title` group.

## Task 24 — Fill Missing Embarked Values 🔄

Fill missing `Embarked` values with an appropriate value.

## Task 25 — Create HasCabin

Create a binary `HasCabin` feature from `Cabin` to indicate whether cabin information is available.

---

# Important Teaching Rule

Do not jump ahead to Machine Learning yet.

The current focus is learning Data Science and EDA properly.

After Task 25, create the next set of tasks based on the user's progress and gradually move toward:

Data Cleaning → Feature Engineering → Visualization → ML preparation → Model training → Evaluation

Continue from Task 21.
