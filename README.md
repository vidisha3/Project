# 🏀 NBA Player Data Analysis Project

## 📌 Project Explanation  
This project focuses on analyzing and understanding data related to professional basketball players using Python. The dataset contains biographical and career-related information such as:

- Height (`Ht`)
- Weight (`Wt`)
- Playing position (`Pos`)
- Birth date
- College
- Career duration

---

## 🎯 Objectives

### 🧹 1. Preprocessing & Cleaning
- Convert raw fields like height (`6-9`) into numerical format (inches).
- Handle missing values.
- Encode categorical features for modeling.

### 📊 2. Visualize Player Attributes
- Distribution of player heights and weights.
- Relationship between height and weight.
- Trends in career length by position.

### 📈 3. Understand Key Relationships
- How physical attributes affect:
  - Player position
  - Career longevity

### 🤖 4. Build Predictive Models
Using supervised machine learning regressors to predict attributes (e.g., weight) based on other features.

#### Models Used:
- 🔹 Linear Regression  
- 🌲 Random Forest Regressor  
- 📈 Gradient Boosting Regressor

## 📊 Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to better understand the dataset and reveal patterns or relationships between features. Key steps included:

- 📌 **Distribution Analysis**: Visualized height, weight, and position distributions using histograms and bar plots.
- 📌 **Correlation Heatmap**: Used a heatmap to examine correlations between numerical features like height, weight, and career length.
- 📌 **Scatter Plots**: Analyzed relationships between height and weight, and how they relate to player positions.
- 📌 **Outlier Detection**: Identified unusual entries or extreme values that could affect model performance.

These insights guided preprocessing choices and helped select relevant features for modeling.


---

## 🧰 Tech Stack

| Tool            | Purpose                          |
|-----------------|----------------------------------|
| `Python`        | Programming Language             |
| `Pandas`        | Data Manipulation                |
| `NumPy`         | Numerical Operations             |
| `Matplotlib` / `Seaborn` | Data Visualization  |
| `Scikit-learn`  | Machine Learning Models          |


