import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- Linear Regression with Boston Housing Dataset ---
def linear_regression_boston():
    # Load Boston Housing Dataset
    boston = fetch_openml(name="boston", version=1, as_frame=True)
    X = boston.data.to_numpy() # Convert DataFrame to NumPy array
    y = boston.target.to_numpy() # Convert Series to NumPy array
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, random_state=42)
    # Train a linear regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    # Predictions
    y_pred = lin_reg.predict(X_test)
    # Performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
             color="red", linewidth=2)
    plt.xlabel("True Values (Boston Housing)")
    plt.ylabel("Predicted Values")
    plt.title("Linear Regression on Boston Housing Dataset")
    plt.grid(True)
    plt.show()
    print(f"Linear Regression Results:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

# --- Polynomial Regression with Auto MPG Dataset ---
def polynomial_regression_auto_mpg():
    # Load Auto MPG Dataset
    auto_mpg = fetch_openml(name="autoMpg", version=1, as_frame=True)
    data = auto_mpg.data
    target = auto_mpg.target.astype(float) # Convert target (MPG) to float
    # Remove rows with missing 'horsepower' values from both data and
    # target
    data = data.dropna(subset=["horsepower"])
    target = target.loc[data.index] # Ensure target is aligned with
                                    # filtered data
    # Select feature (Horsepower) and target (MPG)
    X_hp = data[["horsepower"]].astype(float) # Convert horsepower to
                                            # float
    y_mpg = target
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_hp, y_mpg,
                                                        test_size=0.2, random_state=42)
    # Polynomial regression (degree=3)
    poly_features = PolynomialFeatures(degree=3)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    # Train a linear regression model on transformed polynomial features
    lr_poly = LinearRegression()
    lr_poly.fit(X_train_poly, y_train)
    # Predictions
    y_pred_poly = lr_poly.predict(X_test_poly)
    # Performance metrics
    mse_poly = mean_squared_error(y_test, y_pred_poly)
    r2_poly = r2_score(y_test, y_pred_poly)
    # Sort for plotting
    X_test_sorted, y_test_sorted = zip(*sorted(zip(X_test.values.flatten(),
                                                    y_test)))
    y_pred_sorted = lr_poly.predict(poly_features.transform(np.array(X_test_sorted).reshape(-1, 1)))
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color="blue", label="True values",
                alpha=0.6)
    plt.plot(X_test_sorted, y_pred_sorted, color="red", label="Polynomial fit (degree=3)", linewidth=2)
    plt.xlabel("Horsepower")
    plt.ylabel("Miles Per Gallon (MPG)")
    plt.title("Polynomial Regression on Auto MPG Dataset")
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"Polynomial Regression Results (Degree=3):")
    print(f"Mean Squared Error: {mse_poly:.2f}")
    print(f"R^2 Score: {r2_poly:.2f}")

# Run both demonstrations
def run_models():
    linear_regression_boston()
    polynomial_regression_auto_mpg()

# Execute the functions
run_models()