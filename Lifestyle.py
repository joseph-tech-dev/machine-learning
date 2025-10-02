import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("health_lifestyle_dataset.csv")
#print(df.head(5))
#print(df.info())
#print(df.describe())
#print(df.shape)

df = pd.get_dummies(df, drop_first=True)
print(df.info())

# Distribution of Features
# Histogram of Age
plt.hist(df['age'], bins=30, edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
#plt.show()
plt.savefig("histOfAge.png")

# BMI distribution
plt.hist(df['bmi'], bins=30, color="orange", edgecolor='black')
plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Count")
#plt.show()
plt.savefig("BMI_Distribution.png")

# Correlation HeatMap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
#plt.show()
plt.savefig("Heatmap.png")

# Boxplots: Metrics vs. Disease Risk
plt.figure(figsize=(8,6))
df.boxplot(column="bmi", by="disease_risk")
plt.title("BMI by Disease Risk")
plt.suptitle("")
plt.xlabel("Disease Risk")
plt.ylabel("BMI")
#plt.show()
plt.savefig("Metrics&DiseaseRisk.png")


plt.figure(figsize=(8,6))
df.boxplot(column="systolic_bp", by="disease_risk")
plt.title("Systolic BP by Disease Risk")
plt.suptitle("")
plt.xlabel("Disease Risk")
plt.ylabel("Systolic BP")
#plt.show()

# Scatter Plot
plt.figure(figsize=(8,6))
plt.scatter(df["bmi"], df["daily_steps"], c=df["disease_risk"], cmap="viridis", alpha=0.5)
plt.colorbar(label="Disease Risk")
plt.title("BMI vs Daily Steps")
plt.xlabel("BMI")
plt.ylabel("Daily Steps")
#plt.show()
plt.savefig("")


# Categorical Risk Factor's
risk_smoke = df.groupby("smoker")["disease_risk"].mean()
risk_alcohol = df.groupby("alcohol")["disease_risk"].mean()

fig, ax = plt.subplots(1,2, figsize=(10,4))
risk_smoke.plot(kind="bar", ax=ax[0], color="teal")
ax[0].set_title("Average Disease Risk by Smoking")

risk_alcohol.plot(kind="bar", ax=ax[1], color="brown")
ax[1].set_title("Average Disease Risk by Alcohol")
plt.show()

# Age Vs Risk Line Plot
age_risk = df.groupby("age")["disease_risk"].mean()
plt.plot(age_risk.index, age_risk.values)
plt.title("Average Disease Risk by Age")
plt.xlabel("Age")
plt.ylabel("Risk Probability")
plt.show()


# Features defination
X = df.drop(columns=["id", "disease_risk"])
y = df["disease_risk"]

# Features Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=400)

# Model
model = RandomForestClassifier(max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy:  {(accuracy_score(y_test, y_pred))*100}%")
