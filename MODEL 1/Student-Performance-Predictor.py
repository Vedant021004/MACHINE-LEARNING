import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_excel('AI_Placement_Predictor_Dataset.xlsx')
model = LinearRegression()

print(df.head(10))

print(df.info())

print(df.describe())

print(df[df.isnull().sum())]

print(df.drop_duplicates(inplace=True))

print(df.shape)

print(df.columns)

plt.hist(df["Interview_Score"])

plt.figure(figsize=(8,5))

plt.scatter(df["CGPA"], df["Interview_Score"])

plt.title("CGPA vs Interview Score")
plt.xlabel("CGPA")
plt.ylabel("Interview Score")

# plt.show()

X = df.drop(["Interview_Score", "Candidate_ID"], axis=1)

y = df["Interview_Score"]

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)

model.fit(X_train, y_train)

print(" the slope",model.coef_)

print(model.intercept_)

prediction = model.predict(X_test)

print(prediction[:10])

comparison = pd.DataFrame({

    "Actual": y_test,

    "Predicted": prediction

})

print(comparison.head(10))
score = r2_score(y_test, prediction)

print(score)

print("\nEnter Candidate Details\n")

age = int(input("Age: "))
cgpa = float(input("CGPA: "))
python_skill = int(input("Python Skill (1-10): "))
sql_skill = int(input("SQL Skill (1-10): "))
dsa = int(input("DSA Problems Solved: "))
ml = int(input("ML Knowledge (1-10): "))
linux = int(input("Linux Skill (1-10): "))
github = int(input("GitHub Repositories: "))
projects = int(input("Projects Completed: "))
internships = int(input("Internships: "))
hackathons = int(input("Hackathons: "))
certifications = int(input("Certifications: "))
communication = int(input("Communication Score (1-10): "))
aptitude = int(input("Aptitude Score (1-10): "))
resume = int(input("Resume Score: "))
mock = int(input("Mock Interview Score (1-10): "))
leetcode = int(input("LeetCode Problems Solved: "))
opensource = int(input("Open Source PRs: "))

candidate = [[
    age,
    cgpa,
    python_skill,
    sql_skill,
    dsa,
    ml,
    linux,
    github,
    projects,
    internships,
    hackathons,
    certifications,
    communication,
    aptitude,
    resume,
    mock,
    leetcode,
    opensource
]]

prediction = model.predict(candidate)

print("\nPredicted Interview Score:", round(prediction[0],2))
