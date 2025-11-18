from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

def main():
    # Load dataset
    data = fetch_olivetti_faces()
    X = data.data
    y = data.target

    # Train-test split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train model
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Save model
    joblib.dump(clf, "savedmodel.pth")
    print("Model saved as savedmodel.pth")

if __name__ == "__main__":
    main()
