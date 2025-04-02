# Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

def preprocess_data(df):
    """Fill missing values and convert categorical variables."""
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    # For the 'Embarked' column fill with mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Convert categorical variables using one-hot encoding.
    # We use drop_first=True to avoid multicollinearity.
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    return df

if __name__ == "__main__":
    # Load your Titanic dataset
    data = pd.read_csv('titanic_train.csv')
    
    # Select features and target.
    # We ignore columns like 'PassengerId', 'Name', 'Ticket', 'Cabin' here.
    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']
    X = data[features]
    y = data['Survived']
    
    # Preprocess features
    X = preprocess_data(X)
    
    # Split the data for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate model accuracy on the test set
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Test Accuracy: {:.2f}%".format(acc * 100))
    
    # Save the model to disk
    pickle.dump(model, open('titanic_model.pkl','wb'))
    
    print("Model saved as titanic_model.pkl")