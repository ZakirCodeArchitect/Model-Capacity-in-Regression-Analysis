# 🎓 Deep Learning Polynomial Regression Analysis

This project explores the power of **polynomial regression models** in capturing trends within the **Online Shoppers Purchasing Intention Dataset**. By analyzing univariate models of polynomial degrees 1 through 4, we investigate how model complexity (capacity) impacts performance across various training set sizes. Our goal is to understand **overfitting, underfitting,** and **confidence in model predictions** as data volume changes.

## 🌟 Project Objectives

1. **Model Capacity Exploration**: Train polynomial regression models of increasing complexity to find the optimal balance between model flexibility and accuracy.
2. **Training Size Sensitivity**: Examine model performance across training sets of different sizes (100, 1000, 5000, and 8000) and analyze how training size impacts model accuracy and reliability.
3. **Confidence Interval Analysis**: Plot error bars representing 95% confidence intervals for predictions across 40 different training subsets, providing insights into the variability of model predictions.

## 📂 Project Structure

- **`data/`**: Stores the dataset in `.csv` format.
- **`models/`**: Contains code for implementing and training polynomial regression models of different degrees.
- **`scripts/`**: Includes scripts to preprocess data, train models, and visualize results.
- **`notebooks/`**: Jupyter notebooks for an interactive, step-by-step walkthrough of the analysis.
- **`results/`**: Stores generated plots, graphs, and summary statistics.

## 🚀 Setup & Installation

Clone this repository and set up the environment as follows:

```bash
git clone <repo-url>
cd <project-directory>
pip install -r requirements.txt

## 🛠️ Libraries Used

- **NumPy & Pandas**: Essential for data handling and preprocessing.
- **Scikit-Learn**: Used for polynomial regression, model evaluation, and train-test splitting.
- **Matplotlib & Seaborn**: Data visualization libraries, used for generating plots such as confidence intervals and performance graphs.
- **Jupyter Notebook**: Provides an interactive environment for experimentation and documentation.
```

## 🔍 Methodology

### 1. Data Preparation
   - **Data Splitting**: The dataset is split into an 80:20 ratio for training and testing to ensure robust model evaluation.
   - **Training Sets**: Training sets are created with sizes of 100, 1000, 5000, and 8000, each with 40 random subsets. These subsets capture variability and facilitate confidence interval estimation.

### 2. Model Training
   - **Polynomial Degrees**: Polynomial regression models of degrees 1 (linear) through 4 are implemented to examine different levels of model capacity.
   - **Hyperparameter Tuning**: Optimal model parameters are adjusted based on training performance and error minimization.

### 3. Confidence Interval Calculation
   - For each model and training subset, predictions are calculated, and a 95% confidence interval is generated. This process helps in understanding prediction stability across different training sizes.

### 4. Capacity vs. Training Size Analysis
   - Results are analyzed to determine the relationship between **training size** and **optimal model capacity**. Insights from this analysis help evaluate which model complexity best generalizes across various data volumes.

---

## 📊 Results & Insights

- **Model Complexity Analysis**: Higher-degree polynomials (3 and 4) offer greater flexibility but may risk overfitting on smaller training sets.
- **Confidence Intervals**: Lower-degree models produce narrower confidence intervals, indicating more stable predictions, while higher-degree models show broader intervals, reflecting greater variability.
- **Training Size vs. Model Capacity**: Larger training sizes allow for more complex models without overfitting, enhancing generalization.

---

## 📈 Visualizations

- **Error Bar Plots**: 95% confidence intervals illustrate the range of predictions for each model and training subset.
- **Training Size vs. Optimal Capacity Graph**: This graph demonstrates how the model's optimal complexity evolves with training size.

---

## 📝 Conclusion

This project provides valuable insights into the trade-offs between **model complexity** and **training data volume**. By analyzing confidence intervals and optimal capacity, we gain a better understanding of how to select the best model for a given dataset, balancing between underfitting and overfitting.
