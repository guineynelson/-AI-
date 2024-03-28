# -AI-
スマートファクトリーは、エッジコンピューティングと機械学習を活用して、製造工程の自動化と最適化を図り、生産性の向上と品質管理の強化を実現します。
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Simulated dataset generation
# Features represent various metrics in the manufacturing process (temperature, pressure, etc.)
# Target represents product quality: 1 for pass, 0 for fail
np.random.seed(42)
X = np.random.rand(1000, 5)  # 1000 samples, 5 features
y = (np.sum(X, axis=1) > 2.5).astype(int)  # Arbitrary condition for pass/fail

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Machine Learning Model: Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')

# Edge computing simulation: making a prediction for a new product sample
new_sample = np.random.rand(1, 5)
prediction = model.predict(new_sample)
result = "Pass" if prediction[0] == 1 else "Fail"
print(f'New product quality prediction: {result}')
