# AI/ML Practice Projects

Machine learning projects built while learning Python, NumPy, Pandas, Matplotlib and scikit-learn.

## Projects

### 1. Titanic Survival Prediction (`titanic.py`)
Predicts whether a passenger survived the Titanic using a Random Forest.

- Cleaned the data: filled missing Age with the average, dropped columns with too many missing values, converted Sex and Embarked into 0/1 columns
- Model: Random Forest (100 trees)
- Result: about 81% accuracy (average of 5-fold cross-validation)
- Most important features: Sex, Fare and Age
- Note: training accuracy was 98.6% but test accuracy was 75%, which shows overfitting, so I used cross-validation for a more honest score

### 2. Store Profit Prediction (`project2.py`)
Predicts the profit of general store items using Linear Regression.

- Features: Quantity_Sold, Sale_Price, Expense
- Note: in this small dataset Profit = Sale_Price - Expense, so the model scores almost perfectly. This project was for practice.

## How to run
1. Install the libraries: `pip install pandas scikit-learn matplotlib`
2. Keep the CSV file in the same folder as the .py file
3. Run: `python titanic.py`

## Tools used
Python, Pandas, NumPy, Matplotlib, scikit-learn
