import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
titanic_data = pd.read_csv(r"C:\Users\sumedh hajare\Downloads\Titanic-Dataset.csv")

# Display basic information about the dataset
print("Dataset shape:", titanic_data.shape)
print("\nFirst few rows of the dataset:")
print(titanic_data.head())

# Analyze survival rates
survival_rate = titanic_data['Survived'].mean() * 100
print(f"\nOverall survival rate: {survival_rate:.2f}%")

# Age analysis
print("\nAge statistics:")
print(titanic_data['Age'].describe())

plt.figure(figsize=(10, 6))
sns.histplot(data=titanic_data, x='Age', hue='Survived', multiple='stack', bins=20)
plt.title('Age Distribution by Survival Status')
# Modify the legend labels
plt.legend(labels=['Not Survived', 'Survived'])
plt.show()

# Gender analysis
gender_survival = titanic_data.groupby('Sex')['Survived'].mean().sort_values(ascending=False)
print("\nSurvival rate by gender:")
print(gender_survival)

plt.figure(figsize=(8, 6))
sns.countplot(data=titanic_data, x='Sex', hue='Survived')
plt.title('Gender Distribution by Survival Status')
plt.legend(labels=['Not Survived', 'Survived'])
plt.show()

# Ticket class analysis
class_survival = titanic_data.groupby('Pclass')['Survived'].mean().sort_values(ascending=False)
print("\nSurvival rate by ticket class:")
print(class_survival)

plt.figure(figsize=(8, 6))
sns.countplot(data=titanic_data, x='Pclass', hue='Survived')
plt.title('Ticket Class Distribution by Survival Status')
plt.legend(labels=['Not Survived', 'Survived'])
plt.show()

# Fare analysis
print("\nFare statistics:")
print(titanic_data['Fare'].describe())

plt.figure(figsize=(10, 6))
sns.boxplot(data=titanic_data, x='Survived', y='Fare')
plt.title('Fare Distribution by Survival Status')
plt.show()

# Cabin analysis
titanic_data['Cabin_Known'] = titanic_data['Cabin'].notna().astype(int)
cabin_survival = titanic_data.groupby('Cabin_Known')['Survived'].mean()
print("\nSurvival rate by cabin information:")
print(cabin_survival)

plt.figure(figsize=(8, 6))
sns.countplot(data=titanic_data, x='Cabin_Known', hue='Survived')
plt.title('Cabin Information by Survival Status')
plt.legend(labels=['Not Survived', 'Survived'])
plt.xticks([0, 1], ['Unknown', 'Known'])
plt.show()

# Prepare data for modeling
features = ['Pclass', 'Age', 'Fare', 'Sex', 'Cabin_Known']
X = pd.get_dummies(titanic_data[features], columns=['Sex'])
y = titanic_data['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
survival_model = RandomForestClassifier(random_state=42)
survival_model.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = survival_model.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {model_accuracy:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': survival_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance for Survival Prediction')
plt.show()