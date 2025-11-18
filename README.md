# WineMap AI – Unsupervised Wine Segmentation

End-to-end **unsupervised learning** project to group wines by their **chemical profiles**, so a winery or distributor can manage **quality, pricing, and production** even when human tasting labels aren’t available.

## Project Overview
This project uses the classic **Wine Quality** dataset and clusters wines based only on their physicochemical properties.  
The goal is to find **natural wine segments** (by chemistry, not rating) using a full pipeline:
**EDA → preprocessing → scaling → clustering → dimensionality reduction → evaluation → cluster profiling**.

## What This Shows (Skills)
- Exploratory data analysis (distributions, correlations, outliers)
- Data cleaning & feature preprocessing
- Feature scaling with `StandardScaler`
- Unsupervised modeling with:
  - **K-Means**
  - **Agglomerative (hierarchical) clustering**
  - **Gaussian Mixture Models (GMM)**
- Cluster evaluation using **silhouette score**
- **PCA** for 2D cluster visualization
- Business-style interpretation of clusters (how segments differ chemically)

## Workflow
1. **Load & Explore Data** – inspect wine chemistry features and summary stats.  
2. **EDA** – visualize distributions and correlations; identify skew and outliers.  
3. **Preprocessing & Scaling** – remove the `quality` label from features, scale numeric variables with `StandardScaler`.  
4. **Clustering & Tuning** – run K-Means across multiple k values (elbow + silhouette), compare with hierarchical clustering and a GMM at the best k.  
5. **Evaluation & Visualization** – use silhouette scores, PCA scatterplots, and dendrograms to assess cluster quality.  
6. **Cluster Profiling** – summarize each cluster’s average chemistry (e.g., acidity, alcohol, residual sugar) and relate it to potential quality tiers or pricing bands.

## Key Visuals (from the notebook)

### Wine Feature Distributions / EDA  
![EDA1](img_1.png)

### PCA Projection with Cluster Labels  
![PCA_Clusters](img_2.png)

### Silhouette Score / Elbow Diagnostics  
![Silhouette_Elbow](img_3.png)

### Dendrogram (Hierarchical Clustering)  
![Dendrogram](img_4.png)

## Tech Stack
- **Python**: pandas, numpy  
- **Visualization**: matplotlib, seaborn  
- **ML / Unsupervised**: scikit-learn (KMeans, AgglomerativeClustering, GaussianMixture, PCA), scipy (hierarchical clustering)  
- **Environment**: Jupyter / Google Colab  


