import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # Used to save model, scaler

# Load the dataset (downloaded from UCI)
df = pd.read_csv('student-mat.csv', sep=';')

# Create target label: 'pass' if G3 >= 10, else 'fail'
df['performance'] = df['G3'].apply(lambda x: 'pass' if x >= 10 else 'fail')

# Drop the original grade columns
df = df.drop(columns=['G1', 'G2', 'G3'])

# Encode text (categorical) columns to numbers
label_encoders = {}
for column in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Save encoders for later use

# Split into features (X) and label (y)
X = df.drop('performance', axis=1)
y = df['performance']

# Normalize feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test on test set
y_pred = model.predict(X_test)

# Show results
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save model, scaler, and encoders
joblib.dump(model, 'student_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'encoders.pkl')
print("✅ Model, scaler, and encoders saved.")
