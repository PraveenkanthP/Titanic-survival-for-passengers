import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the Titanic dataset
titanic_df = pd.read_csv("titanic.csv")

# Perform data preprocessing
# (e.g., handle missing values, encode categorical variables)

# Perform feature engineering
# (e.g., create new features, transform existing ones)

# Select features and target variable
X = titanic_df[['Pclass', 'Sex', 'Age', 'Fare']]
y = titanic_df['Survived']

# Encode categorical variables
X = pd.get_dummies(X, columns=['Sex'], drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
