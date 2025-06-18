from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_PATH = Path(__file__).resolve().parent / 'diabetes.csv'


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in zero_not_accepted:
        df[column] = df[column].replace(0, pd.NA)
        df[column] = df[column].fillna(df[column].mean())
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return X, y


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])
    param_grid = {
        'knn__n_neighbors': list(range(1, 31, 2)),
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan', 'minkowski'],
        'knn__p': [1, 2],
        'knn__algorithm': ['auto', 'ball_tree', 'kd_tree']
    }
    search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    print('Best parameters:', search.best_params_)
    print('Best CV accuracy:', search.best_score_)
    y_pred = search.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Test accuracy:', acc)
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    main()
