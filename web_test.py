import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, recall_score
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import streamlit as st
import seaborn as sns


#242 samples for training
#61 samples for test
# Load Data
heart_data = pd.read_csv('heart_disease_data.csv')

# Separate Heart Disease and no heart disease person
with_disease = heart_data[heart_data.target == 1]
without_disease = heart_data[heart_data.target == 0]

# Splitting Feature and Target Individually
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Train Test splitting
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, stratify=Y, random_state=2)

# Train the model
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
svc_clf = SVC(probability=True, random_state=42)

ensemble_clf = VotingClassifier(estimators=[
    ('rf', gb_clf),
    ('svc', svc_clf)
], voting='soft')

ensemble_clf.fit(X_train, Y_train)

# Evaluate the model
train_acc = accuracy_score(ensemble_clf.predict(X_train), Y_train)
test_acc = accuracy_score(ensemble_clf.predict(X_test), Y_test)

# F1 Score
f1 = f1_score(ensemble_clf.predict(X_test), Y_test)

# Precision Score
precision = precision_score(ensemble_clf.predict(X_test), Y_test)

# Confusion matrix
conf_matrix = confusion_matrix(Y_test, ensemble_clf.predict(X_test))

# Recall prediction
recall = recall_score(ensemble_clf.predict(X_test), Y_test)

# Creating the app
st.write("Heart Disease Detection Model")
st.write("Enter the following features to check if the person has Heart Disease or not:")

input_df = st.text_input('Input All the features separated by commas')
input_df_lst = input_df.split(',')

submit = st.button("Submit")

if submit:
    if len(input_df_lst) != X.shape[1]:
        st.write(f"Error: Expected {X.shape[1]} features, but got {len(input_df_lst)} features.")
    else:
        try:
            features = np.array(input_df_lst, dtype=np.float64)
            prediction = ensemble_clf.predict(features.reshape(1, -1))

            if prediction[0] == 0:
                st.write("The person does not have Heart Disease.")
            else:
                st.write("The person has Heart Disease.")
        except ValueError as e:
            st.write(f"Error in input features: {e}")

st.write(f"F1 Score: {f1:.2f}")
st.write(f"Precision: {precision:.2f}")

# Performance Metrics for accuracy, precision, recall, F1 Score
st.subheader("Performance Metrics")
fig_metrics, ax_metrics = plt.subplots()
metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metrics_values = [test_acc, precision, recall, f1]
ax_metrics.bar(metrics_labels, metrics_values, color=['blue', 'green', 'orange', 'red'])
ax_metrics.set_ylabel('Score')
ax_metrics.set_title('Performance Metrics')
st.pyplot(fig_metrics)

# Confusion matrix Graphs
st.subheader("Confusion Matrix")
fig_conf, ax_conf = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_conf)
ax_conf.set_xlabel('Predicted')
ax_conf.set_ylabel('Actual')
ax_conf.set_title('Confusion Matrix')
st.pyplot(fig_conf)