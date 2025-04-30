import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, dendrogram

# Load the CSV
df = pd.read_csv('traffic.csv')

# Remove unnecessary columns
df.drop(['Si. No.', 'Category'], axis=1, inplace=True)

# Helper function for plot conversion
def get_img_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

# Univariate
def univariate_analysis():
    desc = df.describe().T[['mean', 'std']]
    mode = df.mode().iloc[0]
    return desc.to_html(), mode.to_frame().T.to_html()

# Bivariate
def bivariate_analysis():
    visuals = []

    # Pairplot
    sns.pairplot(df.select_dtypes(include=np.number).iloc[:, :5])
    visuals.append(get_img_base64())
    plt.clf()

    # Barplot
    df.iloc[:, :5].mean().plot(kind='bar')
    visuals.append(get_img_base64())
    plt.clf()

    # Boxplot
    sns.boxplot(data=df.select_dtypes(include=np.number).iloc[:, :5])
    visuals.append(get_img_base64())
    plt.clf()

    # Correlation Heatmap
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    visuals.append(get_img_base64())
    plt.clf()

    # Chi-square
    chi2, p, _, _ = chi2_contingency(pd.crosstab(df['State/UT/City'], df['Over Speeding - Cases']))
    return visuals, f"Chi-square = {chi2:.2f}, p = {p:.4f}"

# Multivariate
def multivariate_analysis():
    visuals = []

    num_df = df.select_dtypes(include=np.number).fillna(0)
    X_scaled = StandardScaler().fit_transform(num_df)

    # PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    plt.scatter(components[:, 0], components[:, 1])
    plt.title('PCA Plot')
    visuals.append(get_img_base64())
    plt.clf()

    # KMeans
    km = KMeans(n_clusters=3)
    labels = km.fit_predict(X_scaled)
    plt.scatter(components[:, 0], components[:, 1], c=labels)
    plt.title('KMeans Clusters')
    visuals.append(get_img_base64())
    plt.clf()

    # Hierarchical
    linked = linkage(X_scaled[:50], 'ward')
    dendrogram(linked)
    visuals.append(get_img_base64())
    plt.clf()

    # SVM (binary classification)
    y = (df['Grand Total - Died'] > df['Grand Total - Died'].median()).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)
    model = SVC()
    model.fit(X_train, y_train)
    svm_report = classification_report(y_test, model.predict(X_test))

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_report = classification_report(y_test, nb.predict(X_test))

    # Multi Linear Regression
    reg = LinearRegression()
    y_mlr = df['Grand Total - Cases']
    reg.fit(X_scaled, y_mlr)
    mlr_score = reg.score(X_scaled, y_mlr)

    return visuals, svm_report, nb_report, f"MLR R² = {mlr_score:.4f}"
