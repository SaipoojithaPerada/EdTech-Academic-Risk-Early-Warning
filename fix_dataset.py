import pandas as pd

input_file = "dataset/student_performance.csv"
output_file = "dataset/student_performance_daily.csv"

df = pd.read_csv(input_file)

# Convert the existing study-hours feature into a realistic
# average daily study-hours feature.
# Values are scaled to a 0–10 hours/day range.
df["StudyHours"] = (df["StudyHours"] / 44 * 10).round(1)

# Keep the value within a realistic daily range
df["StudyHours"] = df["StudyHours"].clip(0, 10)

df.to_csv(output_file, index=False)

print("New dataset created successfully!")
print(df["StudyHours"].describe())
print("\nFirst 10 StudyHours/day values:")
print(df["StudyHours"].head(10))