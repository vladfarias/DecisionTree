import os

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Products data

data = {
    'Package_Weight_Gr': [212, 215, 890, 700, 230, 240, 730, 780, 218, 750, 202, 680],
    'Package_Type': ['Cardboard Box', 'Cardboard Box', 'Bubble Wrap', 'Bubble Wrap', 'Cardboard Box', 'Cardboard Box', 'Bubble Wrap', 'Bubble Wrap', 'Cardboard Box', 'Bubble Wrap', 'Cardboard Box', 'Bubble Wrap'],
    'Product_Type': ['Smartphone', 'Tablet', 'Tablet', 'Tablet', 'Smartphone', 'Smartphone', 'Tablet', 'Smartphone', 'Smartphone', 'Tablet', 'Smartphone', 'Tablet']
}

# Creating a DF from data
df = pd.DataFrame(data)

# Separate X (input) and Y (output)
X = df[['Package_Weight_Gr', 'Package_Type']]
y = df['Product_Type']

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)

# Create and fit the transformers on the training data

# Fit the categorical variable 'Package_Type'
le_package_type = LabelEncoder()
le_package_type.fit(X_train['Package_Type'])

# Fit the categorical variable 'Product_Type'
le_product_type = LabelEncoder()
le_product_type.fit(y_train)

# Apply the transformation to the training and testing data of the categorical variable 'Package_Type'
X_train['Package_Type'] = le_package_type.transform(X_train['Package_Type'])
X_test['Package_Type'] = le_package_type.transform(X_test['Package_Type'])

# Apply the transformation to the training and testing data of the categorical variable 'Product_Type'
y_train = le_product_type.transform(y_train)
y_test = le_product_type.transform(y_test)

# Create the model
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Make prediction with the model
y_pred = model.predict(X_test)

# Calculate the accuracy
acc_model = accuracy_score(y_test, y_pred)

# Obtain the classification report
report = classification_report(y_test, y_pred)

# Print
print(f"\nAccuracy: ", round(acc_model,2))

print("\nClassification Report:\n")

print(report)

# Save the trained model and transformers
base_directory = '/home/vladfarias/workspace/personal/dataScience/DSA/developmentML/project-1'
new_directory = 'models-decisionTree'

full_path = os.path.join(base_directory, new_directory)

if not os.path.exists(full_path):
    os.makedirs(full_path)
    
joblib.dump(model, 'models-decisionTree/logistic_model.pkl')
joblib.dump(le_package_type, 'models-decisionTree/transformer_package_type.pkl')
joblib.dump(le_product_type, 'models-decisionTree/transformer_product_type.pkl')
