import os
import pandas as pd
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from joblib import dump

current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, "csv", "insurance.csv")

df = pd.read_csv(csv_path)
df.dropna(axis=0, inplace=True)

x = df["Years of Experience"]
y = df["Salary"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

linear_regressor = LinearRegression()

#Addestramento e predizione
linear_regressor.fit(x_train.to_frame(), y_train)
y_test_pred = linear_regressor.predict(x_test.to_frame())

#Risultati
print("Il mean absolute error è: ", mean_absolute_error(y_test, y_test_pred))
print("Il mean absolute percentage error è: ", mean_absolute_percentage_error(y_test, y_test_pred) * 100)

# Salva il modello
dump(linear_regressor, 'insurance_model_pipeline.joblib')