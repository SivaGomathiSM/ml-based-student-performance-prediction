import joblib
import numpy as np
import pandas as pd

# Load saved model and tools
model = joblib.load('student_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('encoders.pkl')

# Example input (a student) – must be the same number and order of features
# You can get the column list from: print(X.columns.tolist()) during training
# Let's say we had 30 features; provide 30 values
feature_names=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences']
sample_input = [
   0, 1, 0, 2, 0, 1, 0, 1, 1, 2,
    3, 1, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 1, 2, 3, 1]

# Scale input
sample_df=pd.DataFrame([sample_input],columns=feature_names)
sample_input_scaled = scaler.transform(sample_df)

# Predict
prediction = model.predict(sample_input_scaled)
print("🎓 Predicted student performance:", prediction[0])  # Output: pass or fail
