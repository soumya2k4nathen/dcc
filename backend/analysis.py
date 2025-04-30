import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
df = pd.read_csv('traffic.csv')

def perform_univariate():
    result = "<h3>Univariate Analysis</h3>"
    
    # --- Basic Statistics ---
    result += f"<h4>Mean:</h4>{df.mean()}<br>"
    result += f"<h4>Median:</h4>{df.median()}<br>"
    result += f"<h4>Mode:</h4>{df.mode().iloc[0]}<br>"
    result += f"<h4>Standard Deviation:</h4>{df.std()}<br>"
    result += f"<h4>Variance:</h4>{df.var()}<br>"
    result += f"<h4>Skewness:</h4>{df.skew()}<br>"
    result += f"<h4>Kurtosis:</h4>{df.kurtosis()}<br>"

    # --- Histogram Plot ---
    hist_path = 'static/univariate_histogram.png'
    df.hist(bins=20, figsize=(10, 8))
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.clf()

    # --- Box Plot ---
    box_plot_path = 'static/univariate_boxplot.png'
    df.plot(kind='box', figsize=(10, 6))
    plt.title('Box Plot of All Features')
    plt.tight_layout()
    plt.savefig(box_plot_path)
    plt.clf()

    return result, [hist_path, box_plot_path]

def perform_bivariate(x_col, y_col):
    result = "<h3>Bivariate Analysis</h3>"

    # --- Correlation Analysis ---
    corr = df[x_col].corr(df[y_col])
    result += f"<h4>Correlation between {x_col} and {y_col}:</h4> {corr:.4f}<br>"

    # --- Linear Regression ---
    X = df[[x_col]].values
    y = df[y_col].values
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_

    result += f"<h4>Linear Regression Equation:</h4> y = {slope:.2f}x + {intercept:.2f}<br>"

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=x_col, y=y_col, data=df)
    plt.plot(X, model.predict(X), color='red')
    plt.title(f'Linear Regression: {x_col} vs {y_col}')
    lr_path = 'static/linear_regression.png'
    plt.tight_layout()
    plt.savefig(lr_path)
    plt.clf()

    # --- Logistic Regression --- (For classification tasks)
    if df[y_col].nunique() <= 2:  # Check if target column is binary
        model_logreg = LogisticRegression()
        model_logreg.fit(X, y)
        y_pred_logreg = model_logreg.predict(X)
        acc_logreg = accuracy_score(y, y_pred_logreg)
        result += f"<h4>Logistic Regression Accuracy:</h4> {acc_logreg:.4f}<br>"

        # --- Plot Logistic Regression ---
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=x_col, y=y_col, data=df, hue=y_col)
        plt.title(f'Logistic Regression: {x_col} vs {y_col}')
        logreg_path = 'static/logistic_regression.png'
        plt.tight_layout()
        plt.savefig(logreg_path)
        plt.clf()
    else:
        result += "<b>Logistic Regression is not applicable. The target variable is not binary.</b><br>"

    # --- Pair Plot ---
    pair_plot_path = 'static/pair_plot.png'
    sns.pairplot(df)
    plt.title('Pair Plot of All Features')
    plt.tight_layout()
    plt.savefig(pair_plot_path)
    plt.clf()

    # --- Bar Plot ---
    bar_plot_path = 'static/bar_plot.png'
    df[x_col].value_counts().plot(kind='bar', color='skyblue')
    plt.title(f'Bar Plot: {x_col}')
    plt.tight_layout()
    plt.savefig(bar_plot_path)
    plt.clf()

    # --- Box Plot ---
    box_plot_path = 'static/box_plot.png'
    sns.boxplot(x=df[x_col])
    plt.title(f'Box Plot: {x_col}')
    plt.tight_layout()
    plt.savefig(box_plot_path)
    plt.clf()

    # --- Chi-Square Test --- (For categorical features)
    if df[x_col].dtype == 'object' and df[y_col].dtype == 'object':
        contingency_table = pd.crosstab(df[x_col], df[y_col])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        result += f"<h4>Chi-Square Test between {x_col} and {y_col}:</h4> p-value = {p:.4f}<br>"

    # --- SVM --- (only if classification possible)
    possible_targets = [col for col in df.columns if 'target' in col.lower() or df[col].nunique() <= 10 and df[col].dtype != 'float']
    if not possible_targets:
        result += "<b>No valid target found for classification.</b><br>"
        return result, [lr_path, logreg_path, pair_plot_path, bar_plot_path, box_plot_path]

    target_col = possible_targets[0]
    y_raw = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(df[[x_col, y_col]], y_raw, test_size=0.3, random_state=42)

    # Encode categorical target
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # SVM model
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train_encoded)
    y_pred_svm = svm.predict(X_test)
    acc_svm = accuracy_score(y_test_encoded, y_pred_svm)

    result += f"<h4>SVM Accuracy:</h4> {acc_svm:.4f}<br>"

    # --- Plot SVM Decision Boundary ---
    def plot_svm_decision_boundary(model, X, y, save_path):
        h = 0.02
        x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(6, 4))
        plt.contourf(xx, yy, Z, alpha=0.3)
        scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors='k', cmap='Set2')
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.title('SVM Decision Boundary')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.clf()

    svm_path = 'static/svm.png'
    plot_svm_decision_boundary(svm, X_test, y_test_encoded, svm_path)

    return result, [lr_path, logreg_path, pair_plot_path, bar_plot_path, box_plot_path, svm_path]

def perform_multivariate():
    result = "<h3>Multivariate Analysis</h3>"

    # --- PCA ---
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df.select_dtypes(include='number'))
    result += f"<h4>PCA Explained Variance Ratios:</h4> {pca.explained_variance_ratio_}<br>"

    pca_path = 'static/pca.png'
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1])
    plt.title('PCA')
    plt.tight_layout()
    plt.savefig(pca_path)
    plt.clf()

    # --- K-Means Clustering ---
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df.select_dtypes(include='number'))
    df['Cluster'] = kmeans.labels_

    kmeans_path = 'static/kmeans.png'
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=df['Cluster'], palette='Set2')
    plt.title('K-Means Clustering')
    plt.tight_layout()
    plt.savefig(kmeans_path)
    plt.clf()

    # --- Hierarchical Clustering ---
    from scipy.cluster.hierarchy import dendrogram, linkage

    linked = linkage(df.select_dtypes(include='number'), 'single')

    hierarchical_path = 'static/hierarchical.png'
    plt.figure(figsize=(10, 6))
    dendrogram(linked)
    plt.title('Hierarchical Clustering')
    plt.tight_layout()
    plt.savefig(hierarchical_path)
    plt.clf()

    # --- Naive Bayes --- (For classification tasks)
    y_raw = df['target']  # Replace with the actual target column
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['target']), y_raw, test_size=0.3, random_state=42)
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    acc_nb = accuracy_score(y_test, y_pred_nb)

    nb_path = 'static/naive_bayes.png'
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=X_test.iloc[:, 0], y=X_test.iloc[:, 1], hue=y_pred_nb)
    plt.title('Naive Bayes Decision Boundary')
    plt.tight_layout()
    plt.savefig(nb_path)
    plt.clf()

    # --- Multi-Linear Regression ---
    model = LinearRegression()
    X = df.drop(columns=['target'])
    y = df['target']
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)

    result += f"<h4>Multi-Linear Regression Mean Squared Error:</h4> {mse:.4f}<br>"

    mlr_path = 'static/multi_linear_regression.png'
    plt.figure(figsize=(6, 4))
    plt.plot(y, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title('Multi-Linear Regression')
    plt.tight_layout()
    plt.savefig(mlr_path)
    plt.clf()

    return result, [pca_path, kmeans_path, hierarchical_path, nb_path, mlr_path]

# Perform analysis
univariate_result, univariate_plots = perform_univariate()
bivariate_result, bivariate_plots = perform_bivariate('feature1', 'feature2')
multivariate_result, multivariate_plots = perform_multivariate()

# Combine results
all_results = univariate_result + bivariate_result + multivariate_result

# Return results
all_plots = univariate_plots + bivariate_plots + multivariate_plots
