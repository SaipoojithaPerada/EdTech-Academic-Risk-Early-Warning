import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/student_performance_daily.csv")

# Separate input features and target
X = df.drop(["FinalGrade","ExamScore" ],axis=1)
y = df["FinalGrade"]

print("Input features (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
from sklearn.tree import DecisionTreeClassifier

# Create the Decision Tree model
model = DecisionTreeClassifier(
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully!")
# Make predictions on the test data
y_pred = model.predict(X_test)

print("\nFirst 20 actual grades:")
print(y_test.head(20).to_numpy())

print("\nFirst 20 predicted grades:")
print(y_pred[:20])
from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")
from sklearn.metrics import classification_report, confusion_matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
import joblib

# Save the trained model
joblib.dump(model, "model/academic_risk_model.pkl")

print("\nModel saved successfully!")