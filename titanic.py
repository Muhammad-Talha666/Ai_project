import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('titanic.csv')
print(df.shape)
print(df.isnull().sum())


print("Average Age:", df["Age"].mean())

df["Age"] = df["Age"].fillna(df["Age"].mean())

print(df["Age"].isnull().sum())

#drop karna 
df = df.drop(columns=["Cabin","Name","Ticket","PassengerId"])
df= df.dropna(subset="Embarked")
print(df.shape)
print(df.isnull().sum())
print(df.head())


pd.set_option("display.width", 200)
pd.set_option("display.max_columns",20)
df=pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)
print(df.shape)
print(df.columns.tolist())
print(df.head())

X = df.drop(columns=["Survived"])
Y= df["Survived"]
print(X.shape, Y.shape)
X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, Y_train)
print("Forest train:", forest.score(X_train, Y_train), "test:",forest.score(X_test, Y_test))

scores = cross_val_score(forest, X, Y, cv=5)
print("5 scores:", scores)
print("Average:",scores.mean())

importance = pd.Series(forest.feature_importances_,index=X.columns).sort_values(ascending=False)
print(importance)