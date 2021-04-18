# - import libraries 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
# ---------------------------------------------------------

sns.set(style="whitegrid", palette="Set2")

sd = pd.read_csv('stroke-data.csv')
#print(sd.head())
#print(sd.info())
#print(sd.drop(columns = ['id']).describe())
cmap = sns.cm.mako_r
# - round off AGE 
sd['age'] = sd['age'].apply(lambda x : round(x))
sd['bmi'] = sd['bmi'].apply(lambda bmi_value: bmi_value if 12 < bmi_value < 60 else np.nan)

sd.sort_values(['gender', 'age'], inplace = True)
sd.reset_index(drop = True, inplace = True)
sd['bmi'].ffill(inplace = True)

#print (sd.info())

xs = sd['stroke'].value_counts().index
ys = sd['stroke'].value_counts().values

#ax = sns.barplot(xs, ys)
#ax.set_xlabel("Stroke")
#plt.show()

#plt.figure(figsize = (12, 8))
#ax = sns.scatterplot( x = 'bmi', y = 'age', alpha = 0.4, data = sd[sd['stroke'] == 0])
#sns.scatterplot(x = "bmi", y = "age", alpha = 1, data = sd[sd['stroke'] == 1], ax = ax )
#ax.set(xlabel = 'Body Mass Index (BMI)', ylabel = 'Age')
#plt.show()

gender_dict = {'Male': 0, 'Female': 1, 'Other': 2}
residence_type_dict = {'Rural': 0, 'Urban': 1}
ever_married_dict = {'No': 0, 'Yes': 1}
work_type_dict = {'children': 0, 'Never_worked': 1, 'Govt_job': 2, 'Private': 3, 'Self-employed': 4}
smoking_status_dict = {'Unknown': 0, 'never smoked': 1, 'formerly smoked':2, 'smokes': 3}

sd['gender'] = sd['gender'].map(gender_dict)
sd['ever_married'] = sd['ever_married'].map(ever_married_dict)
sd['work_type'] = sd['work_type'].map(work_type_dict)
sd['Residence_type'] = sd['Residence_type'].map(residence_type_dict) 
sd['smoking_status'] = sd['smoking_status'].map(smoking_status_dict)

X = sd.drop(columns=['id', 'stroke'])
y = sd['stroke']

sm = SMOTE(random_state = 2)
X, y = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
pipeline.fit(X_train, y_train)
prediction = pipeline.predict(X_test)

plot_confusion_matrix(pipeline, X_test, y_test, cmap=cmap)
plt.grid(False)
plt.show()
print(f"Accuracy Score : {round(accuracy_score(y_test, prediction) * 100, 2)}%")