import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('project.csv')
print(df.head())
print(df.shape)
print(df.isnull().sum())


plt.bar(df["Item"], df["Profit"])
# plt.scatter(df["Sale_Price"], df["Profit"])
plt.xlabel("Item")
plt.ylabel("Profit")
plt.title("Item wise profit")
plt.xticks(rotation=45)




plt.show()


X =df[['Quantity_Sold', 'Sale_Price', 'Expense']]
Y = df['Profit']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)

print("\n----- Predictions -----")
print(predictions)
print("Actual:", Y_test.values)
print("MSE:", mean_squared_error(Y_test, predictions))
print("Score:", model.score(X_test, Y_test))


new_item = [[8, 1600, 1300]]
new_profit = model.predict(new_item)
print("predicted profit:",new_profit)