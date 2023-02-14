import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load CSV file and filter for specific columns
original_df = pd.read_csv("CarPrice_Assignment.csv")
df = original_df.copy()
print(df.head())
df = df[['horsepower', 'enginesize', 'highwaympg', 'price']]
df['price'] = df['price'] * 2.67
nan_counts = df.isna().sum()

print(nan_counts)
print(df.head())
X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Model
reg = LinearRegression()
reg.fit(X_train, y_train)

r2_score = reg.score(X_test, y_test)
print("R² Score:", r2_score)

# Pickle
pickle.dump(reg, open("model.pkl", "wb"))

