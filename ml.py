# ============================================
# PREDICTIVE ANALYTICS WITH VISUALIZATION
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Metrics
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, log_loss, roc_auc_score
)

import statsmodels.api as sm

# ============================================
# UNIT I: DATA PREPROCESSING
# ============================================

print("\n========== UNIT I ==========\n")

data = {
    'Age': [25, 30, 35, 40, 45],
    'Salary': [30000, 40000, 50000, 60000, 70000],
    'Purchased': [0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Missing values
df.loc[2, 'Salary'] = np.nan
df['Salary'].fillna(df['Salary'].mean(), inplace=True)

# Scaling
scaler = StandardScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

print(df)

# ============================================
# UNIT II: REGRESSION
# ============================================

print("\n========== REGRESSION ==========\n")

X = df[['Age']]
y = df['Salary']

# -------- Linear Regression --------
lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)

# Plot Linear Regression
plt.figure()
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.title("Simple Linear Regression")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()

# -------- Polynomial Regression --------
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

X_range = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
X_range_poly = poly.transform(X_range)
y_poly_pred = poly_model.predict(X_range_poly)

plt.figure()
plt.scatter(X, y)
plt.plot(X_range, y_poly_pred)
plt.title("Polynomial Regression")
plt.show()

# -------- Logistic Regression --------
X_multi = df[['Age', 'Salary']]
y_multi = df['Purchased']

log_reg = LogisticRegression()
log_reg.fit(X_multi, y_multi)

# Decision boundary plot
plt.figure()
plt.scatter(df['Age'], df['Salary'], c=y_multi)

x_min, x_max = df['Age'].min()-1, df['Age'].max()+1
y_min, y_max = df['Salary'].min()-1, df['Salary'].max()+1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()
# ============================================
# SIGMOID CURVE VISUALIZATION (IMPORTANT)
# ============================================

# Use only one feature (Age) to show sigmoid
X_single = df[['Age']]
y_single = df['Purchased']

log_reg_1D = LogisticRegression()
log_reg_1D.fit(X_single, y_single)

# Generate smooth curve
X_range = np.linspace(X_single.min(), X_single.max(), 100).reshape(-1,1)
y_prob = log_reg_1D.predict_proba(X_range)[:,1]

plt.figure()
plt.scatter(X_single, y_single)  # actual data
plt.plot(X_range, y_prob)        # sigmoid curve
plt.title("Sigmoid Curve (Logistic Regression)")
plt.xlabel("Age")
plt.ylabel("Probability")
plt.show()

# -------- OLS --------
X_ols = sm.add_constant(df[['Age']])
ols_model = sm.OLS(y, X_ols).fit()
print(ols_model.summary())

# -------- Metrics --------
print("\nRegression Metrics:")
print("MAE:", mean_absolute_error(y, y_pred))
print("MSE:", mean_squared_error(y, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))
print("R2:", r2_score(y, y_pred))


# ============================================
# UNIT III: CLASSIFICATION
# ============================================

print("\n========== CLASSIFICATION ==========\n")

X = df[['Age', 'Salary']]
y = df['Purchased']

# -------- Models --------
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(kernel='linear', probability=True)
}

for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)

    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y, y_pred))

    # Plot decision boundary
    plt.figure()
    plt.scatter(df['Age'], df['Salary'], c=y)

    xx, yy = np.meshgrid(np.linspace(X['Age'].min()-1, X['Age'].max()+1, 100),
                         np.linspace(X['Salary'].min()-1, X['Salary'].max()+1, 100))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.title(f"{name} Decision Boundary")
    plt.xlabel("Age")
    plt.ylabel("Salary")
    plt.show()

# -------- Confusion Matrix --------
y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------- Additional Metrics --------
y_prob = model.predict_proba(X)[:, 1]

print("\nFinal Model Metrics:")
print("Precision:", precision_score(y, y_pred))
print("Recall:", recall_score(y, y_pred))
print("F1:", f1_score(y, y_pred))
print("Log Loss:", log_loss(y, y_prob))
print("AUC:", roc_auc_score(y, y_prob))

print("\n========== END ==========")