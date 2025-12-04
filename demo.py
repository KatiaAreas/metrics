

import numpy as np
import trimesh
from point_cloud_metrics import PointCloudMetrics, DepthMapMetrics


def create_demo_stl(filepath: str = "demo_cube.stl"):
    """Crée un simple cube STL pour les tests"""
    mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    mesh.export(filepath)
    print(f"✓ Modèle STL de démo créé : {filepath}")
    return filepath


def generate_demo_ply_points(n_expected_points: int = 18, 
                             n_frames: int = 50,
                             noise_std: float = 0.002):
    """
    Génère un nuage de points simulant des détections accumulées
    
    Args:
        n_expected_points: Nombre de points à détecter (18 dans votre cas)
        n_frames: Nombre de frames d'accumulation
        noise_std: Écart-type du bruit gaussien (en mètres)
    """
    np.random.seed(42)
    
    # Positions vraies des 18 points (sur la surface d'un cube virtuel)
    true_positions = np.random.rand(n_expected_points, 3) * 0.08 + 0.01
    
    # Accumuler les détections avec bruit
    accumulated_points = []
    for frame in range(n_frames):
        for pos in true_positions:
            # Ajouter du bruit gaussien
            noisy_point = pos + np.random.normal(0, noise_std, 3)
            accumulated_points.append(noisy_point)
    
    accumulated_points = np.array(accumulated_points)
    
    print(f"✓ Nuage de points généré : {len(accumulated_points)} points")
    print(f"  - Points attendus : {n_expected_points}")
    print(f"  - Frames accumulées : {n_frames}")
    print(f"  - Bruit (std) : {noise_std*1000:.2f} mm")
    
    return accumulated_points


def generate_demo_depth_map(height: int = 480, 
                            width: int = 640,
                            base_depth: float = 1000.0,
                            noise_std: float = 5.0):
    """
    Génère une carte de profondeur synthétique
    
    Args:
        height, width: Dimensions de l'image
        base_depth: Profondeur de base en mm
        noise_std: Bruit en mm
    """
    np.random.seed(42)
    
    # Créer un depth map avec une surface plane inclinée
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Plan incliné
    depth_map = base_depth + 0.1 * x - 0.2 * y
    
    # Ajouter du bruit
    depth_map += np.random.normal(0, noise_std, (height, width))
    
    # Créer quelques pixels invalides (zones sans profondeur)
    invalid_mask = np.random.rand(height, width) < 0.05
    depth_map[invalid_mask] = 0
    
    # Ajouter un "objet" (une bosse)
    center_y, center_x = height // 2, width // 2
    y_grid, x_grid = np.ogrid[:height, :width]
    distance_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    bump = 50 * np.exp(-distance_from_center**2 / (2 * 50**2))
    depth_map -= bump
    
    depth_map = depth_map.astype(np.float32)
    
    print(f"✓ Carte de profondeur générée : {height}x{width}")
    print(f"  - Profondeur moyenne : {np.mean(depth_map[depth_map>0]):.1f} mm")
    print(f"  - Pixels valides : {np.sum(depth_map>0)} / {depth_map.size}")
    
    return depth_map


def demo_1_ply_comparison():
    """Démo complète : Comparaison PLY vs STL"""
    print("\n" + "="*80)
    print("DÉMO 1 : COMPARAISON NUAGE DE POINTS PLY")
    print("="*80 + "\n")
    
    # Créer les données de test
    stl_path = create_demo_stl("demo_cube.stl")
    ply_points = generate_demo_ply_points(
        n_expected_points=18,
        n_frames=50,
        noise_std=0.002  # 2mm de bruit
    )
    
    print("\n" + "-"*80)
    print("Analyse en cours...")
    print("-"*80 + "\n")
    
    # Initialiser les métriques
    metrics = PointCloudMetrics(ply_points, stl_path)
    
    # 1. Clustering
    clusters = metrics.cluster_points(eps=0.005, min_samples=3)
    print(f"Clusters trouvés : {len(clusters)} (attendu : 18)")
    
    # 2. Dispersion
    dispersion = metrics.compute_cluster_dispersion(clusters)
    
    print("\nDispersion des 5 premiers clusters :")
    for cluster_id in list(dispersion.keys())[:5]:
        disp = dispersion[cluster_id]
        print(f"  Cluster {cluster_id}:")
        print(f"    Points : {clusters[cluster_id]['size']}")
        print(f"    Dispersion moyenne : {disp['mean_distance']*1000:.2f} mm")
        print(f"    Axes ellipse : [{disp['ellipse_axes'][0]*1000:.2f}, "
              f"{disp['ellipse_axes'][1]*1000:.2f}, "
              f"{disp['ellipse_axes'][2]*1000:.2f}] mm")
    
    # 3. Distance au STL
    distance_metrics = metrics.compute_point_to_surface_metrics(clusters)
    
    print(f"\nDistance au modèle STL :")
    print(f"  RMSE : {distance_metrics['rmse']*1000:.3f} mm")
    print(f"  MAE : {distance_metrics['mae']*1000:.3f} mm")
    print(f"  Distance moyenne : {distance_metrics['mean_distance']*1000:.3f} mm")
    
    # 4. Métrique combinée
    combined = metrics.compute_combined_metric(clusters, dispersion, 
                                               distance_metrics, alpha=0.5)
    print(f"\nMétrique combinée : {combined*1000:.3f} mm")
    
    # 5. Rapport et visualisation
    print("\n" + "-"*80)
    print("Génération des sorties...")
    print("-"*80)
    
    report = metrics.generate_report(clusters)
    with open("demo_rapport_ply.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("✓ Rapport sauvegardé : demo_rapport_ply.txt")
    
    metrics.visualize_clusters(clusters, save_path="demo_clusters.png")
    print("✓ Visualisation sauvegardée : demo_clusters.png")
    
    return metrics, clusters


def demo_2_depth_comparison():
    """Démo complète : Comparaison depth map vs STL"""
    print("\n" + "="*80)
    print("DÉMO 2 : COMPARAISON CARTE DE PROFONDEUR")
    print("="*80 + "\n")
    
    # Créer les données de test
    stl_path = create_demo_stl("demo_cube_depth.stl")
    depth_map = generate_demo_depth_map(
        height=480,
        width=640,
        base_depth=1000.0,
        noise_std=5.0
    )
    
    # Matrice intrinsèque (exemple RealSense D435)
    K = np.array([
        [615.0, 0, 320.0],
        [0, 615.0, 240.0],
        [0, 0, 1]
    ])
    
    # Matrice de transformation
    RT = np.eye(4)
    RT[:3, 3] = [0, 0, 1]  # 1m de translation en Z
    
    print("\n" + "-"*80)
    print("Analyse en cours...")
    print("-"*80 + "\n")
    
    # Initialiser les métriques
    metrics = DepthMapMetrics(
        depth_map=depth_map,
        K=K,
        RT=RT,
        stl_path=stl_path,
        depth_scale=1000.0
    )
    
    # 1. Déprojection
    point_cloud = metrics.deproject_depth_to_3d()
    print(f"Nuage 3D généré : {len(point_cloud)} points")
    
    # 2. Précision
    accuracy = metrics.compute_accuracy_metrics()
    
    print(f"\nMétriques de précision :")
    print(f"  RMSE : {accuracy['rmse']*1000:.3f} mm")
    print(f"  MAE : {accuracy['mae']*1000:.3f} mm")
    print(f"  Distance moyenne : {accuracy['mean_distance']*1000:.3f} mm")
    print(f"  Médiane : {accuracy['median_distance']*1000:.3f} mm")
    print(f"  95e percentile : {accuracy['percentile_95']*1000:.3f} mm")
    print(f"  Max : {accuracy['max_distance']*1000:.3f} mm")
    
    # 3. Complétude
    completeness = metrics.compute_completeness(threshold=0.01)
    
    print(f"\nComplétude (couverture surface) :")
    print(f"  {completeness['completeness_percentage']:.2f}% de la surface couverte")
    print(f"  Seuil utilisé : {completeness['threshold_m']*1000:.1f} mm")
    
    # 4. Métriques par région
    h, w = depth_map.shape
    regions = [
        (slice(0, h//2), slice(0, w//2)),
        (slice(0, h//2), slice(w//2, w)),
        (slice(h//2, h), slice(0, w//2)),
        (slice(h//2, h), slice(w//2, w))
    ]
    
    regional = metrics.compute_regional_metrics(regions)
    
    print(f"\nMétriques par quadrant :")
    quadrant_names = ["Haut-Gauche", "Haut-Droite", "Bas-Gauche", "Bas-Droite"]
    for i, (name, region_metrics) in enumerate(zip(quadrant_names, regional.values())):
        print(f"  {name} : RMSE = {region_metrics['rmse']*1000:.3f} mm, "
              f"Points = {region_metrics['num_points']}")
    
    # 5. Sorties
    print("\n" + "-"*80)
    print("Génération des sorties...")
    print("-"*80)
    
    report = metrics.generate_report()
    with open("demo_rapport_depth.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("✓ Rapport sauvegardé : demo_rapport_depth.txt")
    
    error_map = metrics.create_error_heatmap(save_path="demo_heatmap.png")
    print("✓ Carte de chaleur sauvegardée : demo_heatmap.png")
    
    return metrics


def main():
    """Programme principal de démonstration"""
    print("\n" + "="*80)
    print("DÉMONSTRATION - TOOLKIT DE MÉTRIQUES DE NUAGES DE POINTS")
    print("="*80)
    print("\nCe script génère des données synthétiques et exécute les analyses complètes.")
    print("Vous obtiendrez :")
    print("  - Rapports texte détaillés")
    print("  - Visualisations 3D")
    print("  - Cartes de chaleur")
    print("\nFichiers générés dans le répertoire courant.")
    
    input("\nAppuyez sur Entrée pour lancer la démo 1 (PLY)...")
    metrics_ply, clusters = demo_1_ply_comparison()
    
    input("\nAppuyez sur Entrée pour lancer la démo 2 (Depth Map)...")
    metrics_depth = demo_2_depth_comparison()
    
    print("\n" + "="*80)
    print("DÉMONSTRATION TERMINÉE")
    print("="*80)
    print("\nFichiers générés :")
    print("  ✓ demo_cube.stl")
    print("  ✓ demo_cube_depth.stl")
    print("  ✓ demo_rapport_ply.txt")
    print("  ✓ demo_clusters.png")
    print("  ✓ demo_rapport_depth.txt")
    print("  ✓ demo_heatmap.png")
    
    print("\nVous pouvez maintenant :")
    print("  1. Consulter les rapports (.txt)")
    print("  2. Voir les visualisations (.png)")
    print("  3. Adapter le code pour vos propres données")
    
    print("\nConsultez README.md et QUICKSTART.py pour plus d'informations !")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
