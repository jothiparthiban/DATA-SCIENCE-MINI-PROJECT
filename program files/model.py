import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, plot_importance

# Load dataset
df = pd.read_csv("C:/new mini/xAPI-Edu-Data.csv")

# Feature engineering function
def feature_engineer(df):
    df['gender'] = df['gender'].map({'M': 0, 'F': 1})
    df['NationalITy'] = LabelEncoder().fit_transform(df['NationalITy'])
    df['StageID'] = df['StageID'].map({'lowerlevel': 0, 'MiddleSchool': 1, 'HighSchool': 2})
    df['GradeID'] = df['GradeID'].map({'G-02': 0, 'G-04': 1, 'G-05': 2, 'G-06': 3, 'G-07': 4,
                                       'G-08': 5, 'G-09': 6, 'G-10': 7, 'G-11': 8, 'G-12': 9})
    df['SectionID'] = df['SectionID'].map({'A': 0, 'B': 1, 'C': 2})
    df['Topic'] = LabelEncoder().fit_transform(df['Topic'])
    df['Semester'] = df['Semester'].map({'F': 0, 'S': 1})
    df['Relation'] = df['Relation'].map({"Father": 0, "Mum": 1})
    df['ParentAnsweringSurvey'] = df['ParentAnsweringSurvey'].map({"No": 0, "Yes": 1})
    df['ParentschoolSatisfaction'] = df['ParentschoolSatisfaction'].map({'Bad': 0, 'Good': 1})
    df['StudentAbsenceDays'] = df['StudentAbsenceDays'].map({'Under-7': 0, 'Above-7': 1})
    df['Class'] = df['Class'].map({"L": 0, "M": 1, "H": 2})
    return df

# Split dataset into train and test sets
df['id_student'] = [f"ID_{i}" for i in range(len(df))]
test = df.sample(n=100).reset_index(drop=True)
train = df[~df['id_student'].isin(test['id_student'].unique())].reset_index(drop=True)
df_train = feature_engineer(train)
df_test = feature_engineer(test)
features = [x for x in df_train.columns if x not in ['PlaceofBirth', 'Class', 'id_student']]

# Train-test split within the training data
X_train, X_valid, y_train, y_valid = train_test_split(df_train[features], df_train['Class'], 
                                                      random_state=42, stratify=df_train['Class'])

# Compute sample weights for class imbalance
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Train the XGBClassifier
xgb_clf = XGBClassifier(objective='multi:softmax', num_class=3, gamma=0, learning_rate=0.1, 
                        max_depth=5, reg_lambda=2, reg_alpha=2, subsample=0.8, 
                        colsample_bytree=0.6, early_stopping_rounds=50, eval_metric=['merror', 'mlogloss'], seed=42)
xgb_clf.fit(X_train, y_train, verbose=1, sample_weight=sample_weights, eval_set=[(X_train, y_train), (X_valid, y_valid)])

# Plot evaluation metrics
results = xgb_clf.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Dev')
ax.legend()
plt.ylabel('mlogloss')
plt.title('XGBoost mlogloss')
plt.show()

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Dev')
ax.legend()
plt.ylabel('merror')
plt.title('XGBoost merror')
plt.show()

plot_importance(xgb_clf)

# Model evaluation
y_pred = xgb_clf.predict(X_valid)
print('\n------------------ Confusion Matrix -----------------\n')
print(confusion_matrix(y_valid, y_pred))

print('\n-------------------- Key Metrics --------------------')
print(f'\nAccuracy: {accuracy_score(y_valid, y_pred):.2f}')
print(f'Balanced Accuracy: {balanced_accuracy_score(y_valid, y_pred):.2f}\n')
print(f'Micro Precision: {precision_score(y_valid, y_pred, average="micro"):.2f}')
print(f'Macro Precision: {precision_score(y_valid, y_pred, average="macro"):.2f}')
print(f'Macro Recall: {recall_score(y_valid, y_pred, average="macro"):.2f}\n')
print('\n--------------- Classification Report ---------------\n')
print(classification_report(y_valid, y_pred))

# Final evaluation on test data
y_pred_test = xgb_clf.predict(df_test[features])
print('\n------------------ Confusion Matrix on Test Data -----------------\n')
print(confusion_matrix(df_test['Class'], y_pred_test))
print(f'\nAccuracy on Test Data: {accuracy_score(df_test["Class"], y_pred_test):.2f}')

# Save the model as .h5 in the same directory as the script
import h5py

# Save the model in JSON format
xgb_clf.save_model("xgb_model.json")

# Read the JSON model data
with open("xgb_model.json", "r") as json_file:
    model_json = json_file.read()

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path for the .h5 file in the same directory
h5_file_path = os.path.join(current_dir, "xgb_model.h5")

# Save JSON data to .h5 format in the same directory as the script
with h5py.File(h5_file_path, "w") as h5_file:
    h5_file.create_dataset("xgb_model", data=model_json)

print(f"Model saved as .h5 file successfully at {h5_file_path}")
