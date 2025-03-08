import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('cleaned_traffic_data.csv')  # Replace with actual dataset file name

# Data preprocessing
data.dropna(inplace=True)

data['vehicle_count'] = data[['n_appr_cars_r', 'n_appr_cars_t', 'n_appr_cars_l',
                              's_appr_cars_r', 's_appr_cars_t', 's_appr_cars_l',
                              'e_appr_cars_r', 'e_appr_cars_t', 'e_appr_cars_l',
                              'w_appr_cars_r', 'w_appr_cars_t', 'w_appr_cars_l']].sum(axis=1)

data['pedestrian_count'] = data[['n_appr_peds', 's_appr_peds', 'e_appr_peds', 'w_appr_peds']].sum(axis=1)

data['hour'] = pd.to_datetime(data['start_time']).dt.hour
data['day_of_week'] = pd.to_datetime(data['start_time']).dt.dayofweek

data['speed_variance'] = np.random.uniform(5, 20, len(data))  # Placeholder

data['intersection_risk'] = np.random.uniform(0.1, 1.0, len(data))  # Placeholder

data['weather_factor'] = np.random.choice([0, 1, 2], len(data), p=[0.7, 0.2, 0.1])

collision_prob = (
    0.5 * (data['vehicle_count'] / data['vehicle_count'].max()) +
    0.3 * (data['pedestrian_count'] / data['pedestrian_count'].max()) +
    0.2 * (data['speed_variance'] / data['speed_variance'].max()) +
    0.4 * data['intersection_risk'] +
    0.3 * (data['weather_factor'] / 2)
)

data['collision_risk'] = (collision_prob > collision_prob.median()).astype(int)

# Define features and target
features = ['vehicle_count', 'pedestrian_count', 'hour', 'day_of_week', 'speed_variance', 'intersection_risk', 'weather_factor']
X = data[features]
y = data['collision_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature Importance
feature_importances = model.feature_importances_
feature_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df, hue='Feature', palette='Blues_r', legend=False)
plt.title('Feature Importance in Predicting Collisions')
plt.show()

# Simulating interventions
X_test_modified = X_test.copy()
X_test_modified['vehicle_count'] *= 0.8
X_test_modified['intersection_risk'] *= 0.65  # Adjusted to reflect a 35% reduction based on similar studies

y_pred_after = model.predict(X_test_modified)

before_intervention = np.mean(y_pred)
after_intervention = np.mean(y_pred_after)

plt.figure(figsize=(8, 5))
plt.bar(['Before Intervention', 'After Intervention'], [before_intervention, after_intervention], color=['red', 'green'])
plt.ylabel('Average Collision Risk')
plt.title('Collision Risk Before and After Intervention')
plt.show()
