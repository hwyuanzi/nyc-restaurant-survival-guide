# Presentation Q&A and Rubric Notes

This file is for the team only. It collects likely presentation questions, short rubric-facing explanations, and talking points for the clustering-related pages so the app UI can stay focused on users rather than graders.

## Why the clustering work is meaningful for the rubric

### Algorithm implementation

- The default clustering algorithm is our own NumPy implementation of K-Means in [models/kmeans_scratch.py](models/kmeans_scratch.py), not a thin wrapper around `sklearn.cluster.KMeans`.
- The live clustering pipeline on the GIS Map and PCA Explorer uses that implementation through [utils/clustering.py](utils/clustering.py).
- We can explain the main design choices:
  - `K-Means++` initialization for more stable centroid seeding.
  - Euclidean distance because features are standardized into a shared scale.
  - Multi-start fitting with several random seeds and selection by silhouette score.
  - A stable schema that preserves the requested `K` clusters so the number chosen in the UI matches the number shown in the demo.

### Technical correctness

- Cluster assignments are learned in a shared interpretable feature space, not in the visualization itself.
- The feature space is built from price tier, rating, review volume, health score, cuisine one-hot features, borough one-hot features, and geographic coordinates.
- PCA is only used to project those learned cluster assignments into 3D for inspection.
- GMM and Ward are comparison baselines, not the main algorithm we claim for the from-scratch requirement.

### Interpretability

- Every cluster has a human-readable label derived from centroid behavior relative to the full dataset.
- Every cluster is backed by:
  - centroid drivers,
  - cluster-level summary statistics,
  - prototype restaurants nearest to the centroid,
  - centroid-distance heatmaps and silhouette metrics.

### Application quality

- The user-facing question is meaningful: do there exist stable, interpretable restaurant groups in NYC when we combine public health inspection data with Google Places quality signals?
- The app answers a real exploratory question a user might ask:
  - what kinds of restaurant groups exist,
  - where they are,
  - which group matches my own tastes,
  - which restaurants are representative of each group.

## Recommended 30-60 second explanation

> We cluster restaurants in a shared interpretable feature space built from cuisine, price, rating, review volume, health inspection score, borough, and location. Our default algorithm is a NumPy implementation of K-Means, so this is not just a library wrapper. The GIS Map and PCA Explorer are two views of the same clustering result: the map shows where those groups live geographically, and the PCA view shows how they separate in feature space. To keep the analysis interpretable, each cluster is paired with centroid drivers, summary statistics, and prototype restaurants nearest to the centroid.

## Likely questions and good answers

### Q: What exactly is clustered?

A: Each restaurant is a point in a numeric feature space. The features are price tier, Google rating, review volume, health inspection score, latitude, longitude, cuisine one-hot features, and borough one-hot features. The clustering algorithm groups restaurants with similar values on those attributes.

### Q: Is PCA doing the clustering?

A: No. PCA is only used after clustering to project the learned structure into 3D for visualization. The actual assignments are computed in the shared clustering pipeline before the plot is drawn.

### Q: Why is this interpretable?

A: Because every input feature has a direct real-world meaning. We can also inspect which features differ most between a cluster centroid and the dataset average, and we can show representative restaurants closest to that centroid.

### Q: Why use Euclidean distance?

A: We standardize the feature matrix before clustering, so Euclidean distance is meaningful across mixed numeric dimensions. In K-Means, Euclidean distance is also the natural objective because the algorithm minimizes within-cluster squared distance.

### Q: Why K-Means as the default?

A: It is a core course algorithm, we implemented it ourselves, it is fast enough for the interactive app, and on this dataset it produces compact clusters that are easy to explain with centroids and prototypes.

### Q: Why also show GMM and Ward?

A: They let us compare different geometry assumptions. GMM allows Gaussian components with soft probabilistic structure, while Ward builds clusters by greedily minimizing variance increases during merges. We use them as baselines to justify why our default choice is reasonable.

### Q: How did you choose K?

A: We expose K as a parameter, but the app also provides an automatic suggestion using silhouette score in the same clustering feature space. We use that as a data-driven starting point, then sanity-check that no single catch-all cluster dominates the result and that the labels/prototypes remain interpretable.

### Q: Why not cluster on raw map coordinates only?

A: That would mostly recover neighborhoods, not restaurant types. Our goal is to find meaningful groups defined by cuisine, quality, affordability, and health-related signals, not just spatial proximity.

### Q: Why not use semantic embeddings for clustering?

A: We intentionally moved clustering to interpretable restaurant attributes so we can defend the result in a presentation. Embeddings can be powerful, but they are harder to explain to a grader when asked what a cluster actually means.

### Q: How do you know the clusters are not just visual artifacts?

A: We do not rely on the PCA picture alone. We also report silhouette score, cluster-size balance, centroid distances, and representative restaurants nearest to each centroid. Those provide quantitative and qualitative evidence.

### Q: Why keep small clusters instead of merging them away?

A: The UI should be faithful to the selected value of K. Small clusters can also be useful because they often represent niche or geographically distinct restaurant segments. If a cluster is tiny, we explain it with its centroid drivers and prototype restaurants instead of hiding it through post-processing.

### Q: Are the clusters personalized per user?

A: The clusters themselves are global and stable. User personalization happens afterward through affinity scoring and cluster matching, so the shared cluster structure remains reproducible across users.

## If a TA pushes on "from scratch"

Key point:

- We should explicitly say that the **from-scratch algorithm requirement is satisfied by the default K-Means path**.
- We should also say that **GMM and Ward are included as baselines for comparison**, not as our from-scratch submission.

Short answer:

> Our main clustering algorithm is the K-Means implementation in `models/kmeans_scratch.py`, which we wrote in NumPy. The other algorithms are comparison baselines to study different cluster assumptions, but the from-scratch requirement is satisfied by the default K-Means pipeline used in the app.

## If a TA asks what to click during the demo

Suggested flow:

1. Open `Restaurant Cluster GIS Map`.
2. Show the default K-Means result and the cluster summary cards.
3. Use `Find Optimal K` or the algorithm comparison block to justify the modeling choice.
4. Switch to `PCA Embedding Explorer`.
5. Explain that this is the same clustering result, now projected into 3D.
6. Show feature loadings, cluster evidence, and prototype restaurants.
7. Tie the explanation back to the recommendation page by showing how user taste is mapped onto the stable cluster structure.

## Files to cite during presentation

- [models/kmeans_scratch.py](models/kmeans_scratch.py)
- [utils/clustering.py](utils/clustering.py)
- [app/pages/3_📍_Restaurant_Cluster_Map.py](app/pages/3_📍_Restaurant_Cluster_Map.py)
- [app/pages/4_📊_PCA_Embedding_Explorer.py](app/pages/4_📊_PCA_Embedding_Explorer.py)
- [app/pages/5_🔮_Recommendations.py](app/pages/5_🔮_Recommendations.py)
