import pandas as pd

# Load the new dataset
df = pd.read_csv("dataset/student_performance.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())
print("\nFinalGrade distribution:")
print(df["FinalGrade"].value_counts().sort_index())
print("\nFinalGrade unique values:")
print(sorted(df["FinalGrade"].unique()))
print("\nAverage values by FinalGrade:")
print(
    df.groupby("FinalGrade")[
        ["Attendance", "AssignmentCompletion", "ExamScore", "StudyHours", "StressLevel"]
    ].mean()
)