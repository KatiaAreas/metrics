# Métriques de Comparaison de Nuages de Points et Cartes de Profondeur

Toolkit Python complet pour comparer des nuages de points (PLY) et des cartes de profondeur contre des modèles de référence STL.

## 📋 Table des matières

- [Installation](#installation)
- [Vue d'ensemble](#vue-densemble)
- [Cas d'usage 1 : Comparaison PLY](#cas-1--comparaison-ply-vs-stl)
- [Cas d'usage 2 : Comparaison Depth Map](#cas-2--comparaison-depth-map-vs-stl)
- [Exemples détaillés](#exemples-détaillés)
- [Métriques disponibles](#métriques-disponibles)
- [API Reference](#api-reference)

## 🔧 Installation

```bash
pip install numpy trimesh scipy scikit-learn matplotlib seaborn plyfile
```

### Dépendances

- `numpy` : Calculs numériques
- `trimesh` : Manipulation de meshes 3D et fichiers PLY/STL
- `scipy` : KD-tree pour recherches spatiales
- `scikit-learn` : Clustering (DBSCAN) et PCA
- `matplotlib` & `seaborn` : Visualisations
- `plyfile` : (Optionnel) Lecture alternative de fichiers PLY

## 🎯 Vue d'ensemble

Ce toolkit fournit deux classes principales :

### 1. `PointCloudMetrics`
Compare un nuage de points accumulé (PLY) contre un modèle STL de référence.
- **Use case** : Points détectés manuellement par frame qui s'accumulent dans le temps
- **Problème** : 18 points attendus, mais bruit créant des ellipses/cercles
- **Métriques** : Dispersion des clusters + Distance au modèle STL

### 2. `DepthMapMetrics`
Compare une carte de profondeur (depth map) contre un modèle STL.
- **Use case** : Image de profondeur (n×m) en mm, modèle STL en m
- **Processus** : Déprojection depth → 3D via matrice [K RT]
- **Métriques** : Précision, complétude, cartes de chaleur

---

## 📊 Cas 1 : Comparaison PLY vs STL

### Contexte
Vous détectez 18 points par frame. Ces points s'accumulent dans le temps pour former un fichier PLY. Idéalement, vous devriez avoir 18 clusters bien définis, mais le bruit crée des dispersions (ellipses/cercles) autour de chaque position théorique.

### Métriques calculées

#### 1. **Clustering des points**
```python
clusters = metrics.cluster_points(eps=0.005, min_samples=3)
```
- Regroupe les points qui devraient représenter la même position
- `eps` : distance maximale entre points d'un cluster (en mètres)
- Retourne un dictionnaire avec centroides et points par cluster

#### 2. **Dispersion (quantification du bruit)**
```python
dispersion = metrics.compute_cluster_dispersion(clusters)
```

Pour chaque cluster, calcule :
- **Écart-type** : Dispersion dans chaque direction (x, y, z)
- **Distance moyenne au centroïde** : Mesure du bruit global
- **Distance maximale** : Worst-case du bruit
- **Axes de l'ellipse** (via PCA) : Dimensions de l'ellipse de dispersion
- **Ratio d'ellipse** : ellipse_axes[0] / ellipse_axes[1] (circularité)

#### 3. **Distance au modèle STL**
```python
distance_metrics = metrics.compute_point_to_surface_metrics(clusters)
```

- **RMSE** : Root Mean Square Error des distances
- **MAE** : Mean Absolute Error
- **Distance moyenne/médiane** : Statistiques centrales
- **Min/Max** : Plage des erreurs

#### 4. **Métrique combinée**
```python
combined = metrics.compute_combined_metric(clusters, dispersion, 
                                          distance_metrics, alpha=0.5)
```

Formule : `erreur = α × distance_au_STL + (1-α) × dispersion_moyenne`
- `α = 0.5` : Poids égal entre précision et bruit
- `α = 0.7` : Favorise la précision (distance au STL)
- `α = 0.3` : Favorise la faible dispersion

### Exemple complet

```python
from point_cloud_metrics import PointCloudMetrics
import trimesh

# Charger le nuage de points PLY
ply_mesh = trimesh.load("points_accumules.ply")
ply_points = ply_mesh.vertices

# Initialiser avec le modèle de référence
metrics = PointCloudMetrics(ply_points, "modele_reference.stl")

# 1. Clustering
clusters = metrics.cluster_points(eps=0.005, min_samples=3)
print(f"Clusters trouvés : {len(clusters)} (attendu : 18)")

# 2. Analyser la dispersion
dispersion = metrics.compute_cluster_dispersion(clusters)
for cluster_id, disp in dispersion.items():
    print(f"Cluster {cluster_id}:")
    print(f"  Dispersion moyenne : {disp['mean_distance']*1000:.2f} mm")
    print(f"  Axes ellipse : {disp['ellipse_axes']*1000} mm")

# 3. Distance au STL
distance_metrics = metrics.compute_point_to_surface_metrics(clusters)
print(f"\nRMSE au STL : {distance_metrics['rmse']*1000:.3f} mm")

# 4. Rapport complet
report = metrics.generate_report(clusters)
print(report)

# 5. Visualisation
metrics.visualize_clusters(clusters, save_path="clusters.png")
```

### Sortie typique

```
Clusters trouvés : 18 (attendu : 18)

Cluster 0:
  Dispersion moyenne : 2.34 mm
  Axes ellipse : [3.1, 2.2, 1.8] mm
  
RMSE au STL : 1.56 mm
Métrique combinée : 1.95 mm
```

---

## 🗺️ Cas 2 : Comparaison Depth Map vs STL

### Contexte
Vous avez une carte de profondeur (n×m) en millimètres (valeurs float non normalisées) et un modèle STL en mètres. Vous devez :
1. Déprojeter la depth map en nuage 3D via les matrices [K RT]
2. Comparer le nuage 3D au modèle STL

### Pipeline

```
Depth Map (mm) → [K, RT] → Point Cloud 3D (m) → Comparaison STL
```

### Métriques calculées

#### 1. **Déprojection depth → 3D**
```python
point_cloud = metrics.deproject_depth_to_3d()
```

Processus :
- Filtre les pixels invalides (depth = 0 ou NaN)
- Conversion mm → m via `depth_scale`
- Utilise K (intrinsèques) pour obtenir (x, y, z) en coordonnées caméra
- Applique [R|t] pour passer en coordonnées monde

Formule :
```
x_cam = (u - cx) × z / fx
y_cam = (v - cy) × z / fy
z_cam = z

P_world = R × P_cam + t
```

#### 2. **Métriques de précision**
```python
accuracy = metrics.compute_accuracy_metrics()
```

- **RMSE** : Erreur quadratique moyenne
- **MAE** : Erreur absolue moyenne
- **Percentiles** (95e, 99e) : Distribution des erreurs
- **Ratio pixels valides** : Couverture de l'image

#### 3. **Complétude (coverage)**
```python
completeness = metrics.compute_completeness(threshold=0.01)
```

Calcule le pourcentage de la surface STL qui est couverte par des points du nuage 3D :
- Échantillonne uniformément la surface STL
- Pour chaque point STL, trouve le point le plus proche dans le nuage
- Compte combien sont dans le seuil de distance

Résultat : `85% de la surface couverte à 10mm près`

#### 4. **Carte de chaleur des erreurs**
```python
error_map = metrics.create_error_heatmap(save_path="heatmap.png")
```

Projette les distances point-to-mesh sur l'image depth originale pour visualiser spatialement les erreurs.

#### 5. **Métriques par région**
```python
regions = [
    (slice(0, h//2), slice(0, w//2)),  # Quadrant haut-gauche
    ...
]
regional_metrics = metrics.compute_regional_metrics(regions)
```

Calcule les métriques séparément pour différentes zones de l'image (utile si certaines régions sont critiques).

### Exemple complet

```python
from point_cloud_metrics import DepthMapMetrics
import numpy as np
import cv2

# Charger la carte de profondeur (16-bit)
depth_map = cv2.imread("depth.png", cv2.IMREAD_ANYDEPTH).astype(np.float32)

# Matrice intrinsèque (exemple RealSense D435)
K = np.array([
    [615.0, 0, 320.0],
    [0, 615.0, 240.0],
    [0, 0, 1]
])

# Matrice extrinsèque [R|t]
RT = np.eye(4)
RT[:3, 3] = [0, 0, 1]  # Translation de 1m en Z

# Initialiser
metrics = DepthMapMetrics(
    depth_map=depth_map,
    K=K,
    RT=RT,
    stl_path="modele.stl",
    depth_scale=1000.0  # mm → m
)

# 1. Déprojection
point_cloud = metrics.deproject_depth_to_3d()
print(f"Points générés : {len(point_cloud)}")

# 2. Précision
accuracy = metrics.compute_accuracy_metrics()
print(f"RMSE : {accuracy['rmse']*1000:.2f} mm")
print(f"MAE : {accuracy['mae']*1000:.2f} mm")

# 3. Complétude
completeness = metrics.compute_completeness(threshold=0.01)
print(f"Couverture surface : {completeness['completeness_percentage']:.1f}%")

# 4. Carte de chaleur
error_map = metrics.create_error_heatmap(save_path="heatmap.png")

# 5. Rapport complet
report = metrics.generate_report()
with open("rapport.txt", "w") as f:
    f.write(report)
```

### Sortie typique

```
Points générés : 245,328 / 307,200 pixels
RMSE : 3.45 mm
MAE : 2.78 mm
95e percentile : 8.92 mm
Couverture surface : 87.3%
```

---

## 📈 Métriques disponibles

### Métriques de dispersion (bruit)

| Métrique | Description | Unité |
|----------|-------------|-------|
| `mean_std` | Écart-type moyen dans les 3 dimensions | mm |
| `mean_distance` | Distance moyenne au centroïde | mm |
| `max_distance` | Distance maximale au centroïde | mm |
| `ellipse_axes` | Dimensions de l'ellipse (PCA) | mm |
| `ellipse_ratio` | Ratio axes[0]/axes[1] (circularité) | - |

### Métriques de précision

| Métrique | Description | Formule |
|----------|-------------|---------|
| `RMSE` | Root Mean Square Error | √(Σd²/n) |
| `MAE` | Mean Absolute Error | Σ\|d\|/n |
| `Mean` | Distance moyenne | Σd/n |
| `Median` | Distance médiane | quantile(50%) |
| `Std` | Écart-type | √(Σ(d-μ)²/n) |
| `Percentile 95` | 95% des erreurs sous ce seuil | quantile(95%) |

### Métriques de complétude

| Métrique | Description |
|----------|-------------|
| `completeness_percentage` | % de surface STL couverte |
| `covered_points` | Nombre de points STL couverts |
| `threshold_m` | Seuil de distance utilisé |

---

## 🔬 API Reference

### PointCloudMetrics

```python
PointCloudMetrics(ply_points: np.ndarray, stl_path: str)
```

**Méthodes principales :**

- `cluster_points(eps, min_samples)` : Clustering DBSCAN
- `compute_cluster_dispersion(clusters)` : Calcul dispersion/bruit
- `compute_point_to_surface_metrics(clusters)` : Distance au STL
- `compute_combined_metric(clusters, dispersion, distance, alpha)` : Métrique combinée
- `visualize_clusters(clusters, save_path)` : Visualisation 3D
- `generate_report(clusters)` : Rapport texte complet

### DepthMapMetrics

```python
DepthMapMetrics(depth_map: np.ndarray, K: np.ndarray, 
                RT: np.ndarray, stl_path: str, depth_scale: float)
```

**Méthodes principales :**

- `deproject_depth_to_3d()` : Conversion depth → 3D
- `compute_accuracy_metrics()` : RMSE, MAE, etc.
- `compute_completeness(threshold)` : % couverture surface
- `create_error_heatmap(save_path)` : Carte de chaleur 2D
- `compute_regional_metrics(regions)` : Métriques par zone
- `generate_report()` : Rapport texte complet

---

## 🎨 Visualisations générées

### 1. Clusters PLY (3D scatter + dispersion + distance STL)
![Exemple clusters](clusters_visualization.png)

### 2. Carte de chaleur des erreurs depth map
![Exemple heatmap](error_heatmap.png)

---

## 💡 Conseils d'utilisation

### Ajuster les paramètres de clustering

```python
# Bruit faible (points précis)
clusters = metrics.cluster_points(eps=0.003, min_samples=5)

# Bruit élevé (points dispersés)
clusters = metrics.cluster_points(eps=0.010, min_samples=3)
```

### Filtrer les outliers

```python
# Supprimer les clusters avec trop peu de points
filtered = {k: v for k, v in clusters.items() if v['size'] >= 5}
```

### Seuils de complétude

```python
# Strict : 5mm
completeness_strict = metrics.compute_completeness(threshold=0.005)

# Permissif : 20mm
completeness_loose = metrics.compute_completeness(threshold=0.020)
```

### Analyser des régions spécifiques

```python
# Région centrale (plus importante)
h, w = depth_map.shape
center_region = [(slice(h//4, 3*h//4), slice(w//4, 3*w//4))]
center_metrics = metrics.compute_regional_metrics(center_region)
```

---

## 📝 Format des fichiers

### Entrées acceptées

**PLY** :
- Format binaire ou ASCII
- Minimum : vertices (x, y, z)
- Chargeable via `trimesh` ou `plyfile`

**STL** :
- Format binaire ou ASCII
- Échelle : mètres recommandé

**Depth Map** :
- Array numpy 2D (n×m)
- Type : `float32` ou `float64`
- Unité : millimètres (ou spécifier `depth_scale`)
- Valeurs invalides : 0 ou NaN

**Matrices** :
- K : 3×3 (float)
- RT : 3×4 ou 4×4 (float)

---

## 🐛 Dépannage

### Problème : Trop/pas assez de clusters

**Solution** : Ajuster `eps` dans `cluster_points()`
```python
# Augmenter eps pour fusionner plus de points
clusters = metrics.cluster_points(eps=0.010)

# Réduire eps pour séparer les points
clusters = metrics.cluster_points(eps=0.003)
```

### Problème : Erreur "No points in cluster"

**Cause** : `min_samples` trop élevé

**Solution** :
```python
clusters = metrics.cluster_points(eps=0.005, min_samples=2)
```

### Problème : Depth map vide après déprojection

**Causes possibles** :
1. Mauvaise matrice K (vérifier fx, fy, cx, cy)
2. Mauvais `depth_scale` (vérifier unités)
3. Tous les pixels sont invalides (vérifier depth_map > 0)

**Debug** :
```python
print(f"Pixels non-nuls : {np.sum(depth_map > 0)}")
print(f"Range depth : [{np.min(depth_map[depth_map>0])}, {np.max(depth_map)}]")
```

---

## 📚 Références

**Algorithmes utilisés :**
- DBSCAN clustering : Ester et al. (1996)
- PCA pour ellipses : Pearson (1901)
- Point-to-mesh distance : Trimesh library
- ICP (si nécessaire) : Besl & McKay (1992)

**Métriques standards :**
- ISO 10360 : Spécifications géométriques des CMM
- VDI/VDE 2634 : Imagerie optique 3D

---

## 📄 Licence

MIT License - Libre d'utilisation

## 🤝 Contribution

N'hésitez pas à ouvrir des issues ou proposer des améliorations !
