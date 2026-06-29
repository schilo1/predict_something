import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

os.makedirs('models', exist_ok=True)

df = pd.read_csv('data/dataset.csv')

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.4f}")

joblib.dump(model, 'models/random_forest_model.pkl')

with open('models/metrics.txt', 'w') as f:
    f.write(f"Accurac: {accuracy:.4f}\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, predictions))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, predictions)))

