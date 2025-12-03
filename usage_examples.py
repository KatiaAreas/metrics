"""
Complete Usage Examples for Point Cloud Metrics
Shows how to use the metrics with real PLY and depth map data
"""

import numpy as np
from point_cloud_metrics import PointCloudMetrics, DepthMapMetrics


def example_1_ply_with_real_data():
    """
    Example 1: Analyze PLY point cloud with accumulated detections
    """
    print("\n" + "="*80)
    print("EXEMPLE 1: Analyse d'un nuage de points PLY")
    print("="*80 + "\n")
    
    # Charger votre fichier PLY
    # Option 1: Utiliser trimesh
    import trimesh
    ply_mesh = trimesh.load("votre_fichier.ply")
    ply_points = ply_mesh.vertices
    
    # Option 2: Utiliser plyfile
    # from plyfile import PlyData
    # plydata = PlyData.read("votre_fichier.ply")
    # ply_points = np.vstack([plydata['vertex']['x'], 
    #                         plydata['vertex']['y'], 
    #                         plydata['vertex']['z']]).T
    
    # Option 3: Si vous avez déjà un array numpy
    # ply_points = np.load("points.npy")
    
    # Initialiser les métriques
    stl_path = "modele_reference.stl"
    metrics = PointCloudMetrics(ply_points, stl_path)
    
    # 1. Clustering des points
    # eps = distance maximale entre points d'un même cluster (en mètres)
    # Pour 18 points attendus, ajuster eps selon votre bruit
    clusters = metrics.cluster_points(eps=0.005, min_samples=3)
    
    # 2. Calculer la dispersion (bruit) de chaque cluster
    dispersion = metrics.compute_cluster_dispersion(clusters)
    
    print("Dispersion par cluster:")
    for cluster_id, disp in dispersion.items():
        print(f"  Cluster {cluster_id}:")
        print(f"    Écart-type moyen: {disp['mean_std']*1000:.2f} mm")
        print(f"    Distance max du centroïde: {disp['max_distance']*1000:.2f} mm")
        print(f"    Axes de l'ellipse: {disp['ellipse_axes']*1000} mm")
    
    # 3. Calculer les distances au modèle STL
    distance_metrics = metrics.compute_point_to_surface_metrics(clusters)
    
    print(f"\nDistance des centroïdes au STL:")
    print(f"  RMSE: {distance_metrics['rmse']*1000:.3f} mm")
    print(f"  MAE: {distance_metrics['mae']*1000:.3f} mm")
    print(f"  Distance moyenne: {distance_metrics['mean_distance']*1000:.3f} mm")
    
    # 4. Métrique combinée (distance + dispersion)
    combined = metrics.compute_combined_metric(clusters, dispersion, 
                                               distance_metrics, alpha=0.5)
    print(f"\nMétrique combinée: {combined*1000:.3f} mm")
    
    # 5. Générer un rapport complet
    report = metrics.generate_report(clusters)
    print("\n" + report)
    
    # Sauvegarder le rapport
    with open("rapport_ply.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    # 6. Visualisation
    metrics.visualize_clusters(clusters, save_path="clusters_visualization.png")
    
    return metrics, clusters


def example_2_depth_map_with_real_data():
    """
    Example 2: Analyze depth map and compare to STL
    """
    print("\n" + "="*80)
    print("EXEMPLE 2: Analyse d'une carte de profondeur")
    print("="*80 + "\n")
    
    # Charger votre carte de profondeur
    # Option 1: Image 16-bit (common for depth cameras)
    # import cv2
    # depth_image = cv2.imread("depth.png", cv2.IMREAD_ANYDEPTH)
    # depth_map = depth_image.astype(np.float32)
    
    # Option 2: Fichier numpy
    # depth_map = np.load("depth.npy")
    
    # Option 3: Depuis un fichier texte
    # depth_map = np.loadtxt("depth.txt")
    
    # Exemple avec données simulées
    depth_map = np.random.rand(480, 640) * 1000 + 500
    depth_map[depth_map < 600] = 0  # Pixels invalides
    
    # Matrice intrinsèque de la caméra (à adapter selon votre caméra)
    # Pour une RealSense D435:
    K = np.array([
        [615.0, 0, 320.0],    # fx, 0, cx
        [0, 615.0, 240.0],     # 0, fy, cy
        [0, 0, 1]              # 0, 0, 1
    ])
    
    # Matrice de transformation [R|t]
    # Si la caméra est à l'identité (pas de rotation/translation):
    RT = np.eye(4)
    
    # Si vous avez une rotation et translation:
    # R = rotation_matrix_3x3
    # t = translation_vector_3x1
    # RT = np.eye(4)
    # RT[:3, :3] = R
    # RT[:3, 3] = t
    
    # Initialiser les métriques
    stl_path = "modele_reference.stl"
    metrics = DepthMapMetrics(
        depth_map=depth_map,
        K=K,
        RT=RT,
        stl_path=stl_path,
        depth_scale=1000.0  # mm -> m
    )
    
    # 1. Déprojection depth -> 3D
    point_cloud = metrics.deproject_depth_to_3d()
    print(f"Nuage de points généré: {len(point_cloud)} points")
    
    # 2. Métriques de précision
    accuracy = metrics.compute_accuracy_metrics()
    
    print("\nMétriques de précision:")
    print(f"  RMSE: {accuracy['rmse']*1000:.3f} mm")
    print(f"  MAE: {accuracy['mae']*1000:.3f} mm")
    print(f"  Distance moyenne: {accuracy['mean_distance']*1000:.3f} mm")
    print(f"  95e percentile: {accuracy['percentile_95']*1000:.3f} mm")
    print(f"  Pixels valides: {accuracy['num_valid_points']} / {accuracy['total_pixels']}")
    
    # 3. Complétude (couverture de la surface STL)
    completeness = metrics.compute_completeness(threshold=0.01)
    
    print(f"\nComplétude:")
    print(f"  Couverture de surface: {completeness['completeness_percentage']:.2f}%")
    print(f"  Seuil utilisé: {completeness['threshold_m']*1000:.1f} mm")
    
    # 4. Carte de chaleur des erreurs
    error_map = metrics.create_error_heatmap(save_path="error_heatmap.png")
    
    # 5. Métriques par région (exemple: diviser l'image en 4 quadrants)
    h, w = depth_map.shape
    regions = [
        (slice(0, h//2), slice(0, w//2)),           # Haut gauche
        (slice(0, h//2), slice(w//2, w)),           # Haut droite
        (slice(h//2, h), slice(0, w//2)),           # Bas gauche
        (slice(h//2, h), slice(w//2, w))            # Bas droite
    ]
    
    regional_metrics = metrics.compute_regional_metrics(regions)
    
    print("\nMétriques par région:")
    for region_name, region_metrics in regional_metrics.items():
        print(f"  {region_name}:")
        print(f"    RMSE: {region_metrics['rmse']*1000:.3f} mm")
        print(f"    Points: {region_metrics['num_points']}")
    
    # 6. Rapport complet
    report = metrics.generate_report()
    print("\n" + report)
    
    # Sauvegarder
    with open("rapport_depth.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    return metrics


def example_3_batch_processing():
    """
    Example 3: Process multiple PLY files or depth maps in batch
    """
    print("\n" + "="*80)
    print("EXEMPLE 3: Traitement par lots")
    print("="*80 + "\n")
    
    import glob
    
    # Traiter plusieurs fichiers PLY
    ply_files = glob.glob("data/*.ply")
    stl_reference = "modele_reference.stl"
    
    results = []
    
    for ply_file in ply_files:
        print(f"\nTraitement de {ply_file}...")
        
        import trimesh
        ply_mesh = trimesh.load(ply_file)
        ply_points = ply_mesh.vertices
        
        metrics = PointCloudMetrics(ply_points, stl_reference)
        clusters = metrics.cluster_points(eps=0.005)
        distance_metrics = metrics.compute_point_to_surface_metrics(clusters)
        
        results.append({
            'file': ply_file,
            'n_clusters': len(clusters),
            'rmse': distance_metrics['rmse'],
            'mae': distance_metrics['mae']
        })
    
    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DU TRAITEMENT PAR LOTS")
    print("="*80)
    for result in results:
        print(f"{result['file']}:")
        print(f"  Clusters: {result['n_clusters']}")
        print(f"  RMSE: {result['rmse']*1000:.3f} mm")
        print(f"  MAE: {result['mae']*1000:.3f} mm")


def example_4_advanced_analysis():
    """
    Example 4: Advanced analysis with custom thresholds and filtering
    """
    print("\n" + "="*80)
    print("EXEMPLE 4: Analyse avancée")
    print("="*80 + "\n")
    
    # Charger les données
    import trimesh
    ply_points = trimesh.load("votre_fichier.ply").vertices
    stl_path = "modele_reference.stl"
    
    metrics = PointCloudMetrics(ply_points, stl_path)
    
    # Tester différents paramètres de clustering
    eps_values = [0.003, 0.005, 0.007, 0.010]
    
    print("Test de différents paramètres de clustering:")
    for eps in eps_values:
        clusters = metrics.cluster_points(eps=eps, min_samples=3)
        distance_metrics = metrics.compute_point_to_surface_metrics(clusters)
        
        print(f"\neps={eps*1000:.1f}mm:")
        print(f"  Clusters trouvés: {len(clusters)}")
        print(f"  RMSE: {distance_metrics['rmse']*1000:.3f} mm")
    
    # Filtrage des outliers
    print("\n" + "-"*80)
    print("Filtrage des outliers:")
    
    clusters = metrics.cluster_points(eps=0.005)
    
    # Supprimer les clusters avec trop peu de points
    min_points_per_cluster = 5
    filtered_clusters = {
        k: v for k, v in clusters.items() 
        if v['size'] >= min_points_per_cluster
    }
    
    print(f"  Clusters avant filtrage: {len(clusters)}")
    print(f"  Clusters après filtrage: {len(filtered_clusters)}")
    
    # Recalculer les métriques avec données filtrées
    distance_metrics = metrics.compute_point_to_surface_metrics(filtered_clusters)
    print(f"  RMSE après filtrage: {distance_metrics['rmse']*1000:.3f} mm")


def main():
    """
    Main function showing all examples
    """
    print("\n" + "="*80)
    print("EXEMPLES D'UTILISATION - MÉTRIQUES DE NUAGES DE POINTS")
    print("="*80)
    
    print("\nChoisissez un exemple:")
    print("1. Analyse PLY (points accumulés)")
    print("2. Analyse carte de profondeur")
    print("3. Traitement par lots")
    print("4. Analyse avancée")
    print("5. Tous les exemples")
    
    choice = input("\nVotre choix (1-5): ").strip()
    
    try:
        if choice == "1":
            example_1_ply_with_real_data()
        elif choice == "2":
            example_2_depth_map_with_real_data()
        elif choice == "3":
            example_3_batch_processing()
        elif choice == "4":
            example_4_advanced_analysis()
        elif choice == "5":
            # Exécuter tous les exemples (avec données simulées)
            print("\nExécution de tous les exemples avec données simulées...")
            print("\nPour utiliser avec vos données réelles, modifiez les chemins de fichiers")
            print("dans les fonctions d'exemple.\n")
            
    except FileNotFoundError as e:
        print(f"\nErreur: Fichier non trouvé - {e}")
        print("Veuillez adapter les chemins de fichiers dans le code.")
    except Exception as e:
        print(f"\nErreur: {e}")
        print("Assurez-vous d'avoir installé toutes les dépendances:")
        print("  pip install numpy trimesh scipy scikit-learn matplotlib seaborn")


if __name__ == "__main__":
    main()
