#age,gender,cp,smoker,thalach,BP,cholesterol,glucose,physical_activity,target
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, f1_score, log_loss
from sklearn.impute import SimpleImputer

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'               # Turn off oneDNN custom operations

tf.get_logger().setLevel('ERROR')           # Set TensorFlow logger to only show errors

# Function to load the dataset based on user input
def load_dataset(choice):
    file_paths = [
        'Magister/cardio_train_fixed.csv',
        'Magister/framingham.csv',
        'Magister/heartdisease_fixed2.csv',
        'Magister/Heart_disease_cleveland_new.csv',
        'Magister/heart.csv'
    ]
    if choice < 1 or choice > 5:
        raise ValueError("Invalid choice. Please select a number between 1 and 5.")
    return pd.read_csv(file_paths[choice - 1])

# Function to preprocess the dataset
def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.3, random_state=42)

# Function to build and train the chosen model
def train_model(X_train, y_train, classifier_choice):
    if classifier_choice == 1:
        model = LogisticRegression()
    elif classifier_choice == 2:
        model = RandomForestClassifier()
    elif classifier_choice == 3:
        model = KNeighborsClassifier()
    elif classifier_choice == 4:
        model = SVC(probability=True)
    elif classifier_choice == 5:
        input_dim = X_train.shape[1]
        output_dim = len(np.unique(y_train))
        model = Sequential()
        model.add(Dense(64, input_dim=input_dim, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(output_dim, activation='softmax' if output_dim > 1 else 'sigmoid'))
        model.compile(loss='categorical_crossentropy' if output_dim > 1 else 'binary_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
        y_train = to_categorical(y_train) if output_dim > 1 else y_train
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
        return model
    else:
        raise ValueError("Invalid classifier choice. Please select a number between 1 and 5.")
    
    model.fit(X_train, y_train)
    return model

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Function to plot ROC curve
def plot_roc_curve(y_test, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Main function
if __name__ == "__main__":
    # User input to choose the dataset
    choice = int(input("Choose a dataset (1-5): "))
    df = load_dataset(choice)

    # Define target columns for each dataset
    target_columns = {
        1: 'cardio',                    # cardio_train.csv
        2: 'TenYearCHD',                # framingham.csv
        3: 'num',                       # heartdisease.csv
        4: 'target',                    # Heart_disease_cleveland_new.csv
        5: 'target'                     # heart.csv
    }

    # Preprocess the data
    target_column = target_columns[choice]
    X_train, X_test, y_train, y_test = preprocess_data(df, target_column)

    # User input to choose the classifier
    print("Choose a classifier:")
    print("1. Logistic Regression")
    print("2. Random Forest")
    print("3. k-Nearest Neighbors (kNN)")
    print("4. Support Vector Machines (SVM)")
    print("5. Deep Neural Networks (DNN)")
    classifier_choice = int(input("Enter the number of the classifier (1-5): "))

    # Train the chosen model
    model = train_model(X_train, y_train, classifier_choice)

    # Make predictions
    if classifier_choice == 5:
        y_pred_prob = model.predict(X_test)
        if len(np.unique(y_test)) > 1:
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_pred_prob = y_pred_prob[:, 1]  # Extract probabilities for the positive class
        else:
            y_pred = (y_pred_prob > 0.5).astype(int)
    else:
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

    # Ensure y_test is not one-hot encoded
    if classifier_choice == 5 and len(np.unique(y_test)) > 1:
        y_test = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    logloss = log_loss(y_test, y_pred_prob, labels=np.unique(y_test))

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Log-loss: {logloss:.4f}")

    # Plot ROC curve
    plot_roc_curve(y_test, y_pred_prob)

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=np.unique(y_test))
    plt.show()

    # Print classification report
    print(classification_report(y_test, y_pred))