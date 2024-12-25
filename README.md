
# Remote Monitering System

This project demonstrates a comprehensive machine learning pipeline for data analysis and predictive modeling. It leverages various classifiers and metrics to evaluate performance.

## Dataset

The dataset used for this project is `NIT_dataset.csv`. Ensure the dataset is loaded in the appropriate path before running the script.

## Libraries and Dependencies

The following libraries are used in this project:
- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- xgboost
- lightgbm
- statistics
- warnings

Ensure all these libraries are installed. You can install them using pip:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn xgboost lightgbm
```

## Script Overview

The script performs the following steps:
1. Load the dataset.
2. Display a sample of the dataset for validation.
3. Generate and visualize a correlation matrix.
4. Preprocess the data and split it into training and testing datasets.
5. Train multiple classifiers:
   - Random Forest
   - Logistic Regression
   - Support Vector Classifier (LinearSVC)
   - Naive Bayes
   - Decision Tree
   - XGBoost
   - LightGBM
6. Evaluate the models using metrics such as accuracy, precision, recall, F1 score, log loss, and Matthews correlation coefficient.
7. Implement a Voting Classifier for ensemble learning.

## How to Run the Script

1. Clone the repository or download the script.
2. Place the dataset `NIT_dataset.csv` in the specified directory.
3. Run the script in your Python environment:
   ```bash
   python script_name.py
   ```
   Replace `script_name.py` with the actual script filename.

## Outputs

- Sample data preview.
- Correlation heatmap.
- Performance metrics for each classifier.
- Voting Classifier results.

## Notes

- The dataset should be preprocessed if necessary before running the script.
- Ensure the correct file path is specified for the dataset.


