import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    return (
        KNeighborsClassifier,
        StandardScaler,
        cross_val_score,
        pd,
        plt,
        train_test_split,
    )


@app.cell
def _(StandardScaler, pd, train_test_split):
    df = pd.read_csv('datasets/penguins_size.csv')
    df = df.dropna()
    df = df[df['sex'] != '.']

    X = df.drop('species', axis=1)
    y = df['species']

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X, X_test, X_train, df, scaler, y, y_test, y_train


@app.cell
def _(X_test, X_train, df, y_test, y_train):
    print(f"Original dataset size: {df.shape}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    return


@app.cell
def _(X_train):
    print("First 3 rows of X_train (Standardized):")
    print(X_train[:3])
    return


@app.cell
def _(KNeighborsClassifier, X_test, X_train, y_test, y_train):
    k_candidates = []
    accuracies = []

    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        accuracies.append(acc)
    
        if acc >= 0.85:
            k_candidates.append((k, acc))

    print("K values with accuracy >= 85%:")
    for k, acc in k_candidates:
        print(f"K={k}: {acc:.4f}")
    return (accuracies,)


@app.cell
def _(accuracies, plt):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 21), accuracies, marker='o', linestyle='dashed')
    plt.title('Accuracy vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(KNeighborsClassifier, X, cross_val_score, scaler, y):
    knn_cv = KNeighborsClassifier(n_neighbors=5)

    cv_scores = cross_val_score(knn_cv, scaler.fit_transform(X), y, cv=3)

    print(f"Scores for each block: {cv_scores}")
    print(f"Average Accuracy: {cv_scores.mean():.4f}")
    return


if __name__ == "__main__":
    app.run()
