import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)

clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy: {clf.score(X_test, y_test):.2%}")
print(classification_report(y_test, y_pred, target_names=data.target_names))

print(f"Accuracy: {clf.score(X_test, y_test):.2%}")
print(f"Sample Prediction: {data.target_names[clf.predict(X_test[:1])[0]]}")
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.title("Decision Tree for Breast Cancer Classification")
plt.show()