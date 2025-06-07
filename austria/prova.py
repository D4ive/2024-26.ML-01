import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor


# Construct relative path and load data
df = pd.read_csv("austria\insurance.csv")

# Clean data
df.dropna(axis=0, inplace=True)

# Initial data checks
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isna().sum())
print("\nDataset info:")
print(df.info())

# -----------------------
# Data Splitting
# -----------------------
x_train, x_test, y_train, y_test = train_test_split(
    df.drop(columns=["charges"]),
    df["charges"],
    test_size=0.2,
    random_state=42,
)

# -----------------------
# Model Pipeline
# -----------------------
# Column transformer with OneHotEncoder for categorical variables
encoder = ColumnTransformer(
    [
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist'), 
         ["sex", "smoker", "region"])
    ],
    remainder="passthrough",
    verbose_feature_names_out=False
)

# Random Forest Regressor with reasonable default parameters
regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# Build pipeline
pipe = Pipeline([
    ("encoder", encoder),
    ("standardization", StandardScaler()),
    ("regressor", regressor)
])

# Train and evaluate model
print("\n" + "="*50)
print("TRAINING MODEL")
print("="*50)

pipe.fit(x_train, y_train)
y_test_pred = pipe.predict(x_test)

print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_test_pred):.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mean_absolute_percentage_error(y_test, y_test_pred):.4f}")

# -----------------------
# BMI Analysis
# -----------------------
print("\n" + "="*50)
print("BMI ANALYSIS")
print("="*50)

# Create healthy BMI indicator
df["salutare"] = df["bmi"].between(18.5, 24.9)
print("BMI Health Status:")
print(df["salutare"].value_counts())

# BMI categories function
def categoria_bmi(bmi):
    if bmi < 18.5:
        return "Sottopeso"
    elif bmi < 25:
        return "Normopeso"
    elif bmi < 30:
        return "Sovrappeso"
    else:
        return "Obeso"

# Apply BMI categorization
df["categoria_bmi"] = df["bmi"].apply(categoria_bmi)

print("\nBMI Categories:")
print(df["categoria_bmi"].value_counts())
print("\nSample of BMI data:")
print(df[["bmi", "categoria_bmi", "salutare"]].head(10))

