from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def main():
    # Load dataset
    data = fetch_olivetti_faces()
    X = data.data
    y = data.target

    # Split same way as in train.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Load the trained model
    clf = joblib.load("savedmodel.pth")

    # Predict
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
