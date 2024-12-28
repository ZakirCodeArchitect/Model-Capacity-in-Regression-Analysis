import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('online_shoppers_intention.csv')  # Adjust the path accordingly

# Ensure no NaN values are present
print("NaN values in features:", np.isnan(data['ProductRelated_Duration']).sum())
print("NaN values in target:", np.isnan(data['BounceRates']).sum())

# Remove NaN values if present
data = data.dropna(subset=['ProductRelated_Duration', 'BounceRates'])

# Assign features and target variable
X = data[['ProductRelated_Duration']].values
y = data['BounceRates'].values

# Generate random training sets of specified sizes
sizes = [100, 1000, 5000, 8000]
degrees = [1, 2, 3, 4]
results = {size: {degree: {'mse_train': [], 'mse_test': []} for degree in degrees} for size in sizes}

# Create a function to evaluate the models
def evaluate_models(X_train, y_train, X_test, y_test):
    mse_results = {}
    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly_train = poly_features.fit_transform(X_train)
        X_poly_test = poly_features.transform(X_test)

        # Fit model
        model = LinearRegression()
        model.fit(X_poly_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_poly_train)
        y_test_pred = model.predict(X_poly_test)

        # Calculate MSE
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)

        mse_results[degree] = {
            'mse_train': mse_train,
            'mse_test': mse_test
        }
    
    return mse_results

# Loop through each training size
for size in sizes:
    for _ in range(40):  # Generate 40 random samples for each size
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=np.random.randint(1, 1000))
        mse_results = evaluate_models(X_train, y_train, X_test, y_test)

        # Store the results
        for degree in degrees:
            results[size][degree]['mse_train'].append(mse_results[degree]['mse_train'])
            results[size][degree]['mse_test'].append(mse_results[degree]['mse_test'])

# Prepare data for plotting
mean_mse_train = {size: {degree: np.mean(results[size][degree]['mse_train']) for degree in degrees} for size in sizes}
std_mse_train = {size: {degree: np.std(results[size][degree]['mse_train']) for degree in degrees} for size in sizes}
mean_mse_test = {size: {degree: np.mean(results[size][degree]['mse_test']) for degree in degrees} for size in sizes}
std_mse_test = {size: {degree: np.std(results[size][degree]['mse_test']) for degree in degrees} for size in sizes}

# MSE vs Polynomial Degree for different training sizes
plt.figure(figsize=(14, 8))
for size in sizes:
    plt.errorbar(degrees, [mean_mse_train[size][deg] for deg in degrees],
                 yerr=[std_mse_train[size][deg] for deg in degrees],
                 label=f'Train Size = {size} (Train)', fmt='o', capsize=5)
    
    plt.errorbar(degrees, [mean_mse_test[size][deg] for deg in degrees],
                 yerr=[std_mse_test[size][deg] for deg in degrees],
                 label=f'Train Size = {size} (Test)', fmt='x', linestyle='--', capsize=5)

plt.xlabel('Polynomial Degree', fontsize=14)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.title('MSE vs Polynomial Degree for Different Training Sizes', fontsize=16)
plt.xticks(degrees)
plt.legend()
plt.grid()
plt.show()

# Optimal Capacity Analysis
optimal_capacity = {degree: [] for degree in degrees}
for degree in degrees:
    for size in sizes:
        optimal_capacity[degree].append(np.mean([results[size][degree]['mse_test'][i] for i in range(40)]))

# Plotting Optimal Capacity
plt.figure(figsize=(12, 6))
for degree in degrees:
    plt.plot(sizes, optimal_capacity[degree], marker='o', label=f'Polynomial Degree {degree}')

plt.xscale('log')
plt.xlabel('Training Size (log scale)', fontsize=14)
plt.ylabel('Mean Test MSE', fontsize=14)
plt.title('Optimal Capacity Analysis', fontsize=16)
plt.xticks(sizes, sizes)
plt.legend()
plt.grid()
plt.show()
