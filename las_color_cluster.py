"""
Script to find clusters in LiDAR point cloud based on color and spatial distance.
Each cluster is exported to a separate .las file.
"""

import sys
import os
import numpy as np

try:
    import laspy
except ImportError:
    print("Error: laspy library is not installed.")
    print("Please install it using: pip install laspy")
    sys.exit(1)

try:
    from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install it for better clustering: pip install scikit-learn")


def calculate_color_distance(rgb1, rgb2):
    """
    Calculate Euclidean distance between two RGB colors.
    
    Args:
        rgb1: First RGB color (array or tuple)
        rgb2: Second RGB color (array or tuple)
        
    Returns:
        float: Color distance
    """
    return np.sqrt(np.sum((rgb1 - rgb2) ** 2))


def cluster_by_color(las_file_path, n_clusters=100, method='kmeans', eps_color=30, min_samples=10, 
                    output_dir=None, max_points=200000):
    """
    Cluster points based on color similarity for object extraction.
    
    Args:
        las_file_path (str): Path to the input .las file
        n_clusters (int): Number of clusters to create (for K-means, default: 100)
        method (str): Clustering method ('kmeans', 'minibatch', or 'dbscan')
        eps_color (float): Maximum color distance for clustering (for DBSCAN)
        min_samples (int): Minimum points per cluster (for DBSCAN)
        output_dir (str): Directory to save output files (optional)
        max_points (int): Maximum points to process (for large files)
        
    Returns:
        list: List of output file paths
    """
    if not os.path.exists(las_file_path):
        raise FileNotFoundError(f"File not found: {las_file_path}")
    
    if not las_file_path.lower().endswith('.las'):
        raise ValueError(f"File must be a .las file: {las_file_path}")
    
    print(f"Reading LiDAR data from: {las_file_path}")
    las = laspy.read(las_file_path)
    
    # Get coordinates
    x_coords = np.array(las.x)
    y_coords = np.array(las.y)
    z_coords = np.array(las.z)
    
    total_points = len(x_coords)
    print(f"  Total points: {total_points:,}")
    
    # Check if RGB data exists
    has_rgb = hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue')
    
    if not has_rgb:
        print("  Warning: No RGB color data found in LAS file.")
        print("  Clustering will be based on spatial distance only.")
        red = np.zeros(total_points, dtype=np.uint16)
        green = np.zeros(total_points, dtype=np.uint16)
        blue = np.zeros(total_points, dtype=np.uint16)
    else:
        red = np.array(las.red)
        green = np.array(las.green)
        blue = np.array(las.blue)
        print(f"  RGB data found: {red.min()}-{red.max()}, {green.min()}-{green.max()}, {blue.min()}-{blue.max()}")
    
    # Downsample if too many points (reduce default for memory efficiency)
    if total_points > max_points:
        print(f"  Downsampling from {total_points:,} to {max_points:,} points for clustering...")
        sample_indices = np.random.choice(total_points, max_points, replace=False)
        x_coords = x_coords[sample_indices]
        y_coords = y_coords[sample_indices]
        z_coords = z_coords[sample_indices]
        red = red[sample_indices]
        green = green[sample_indices]
        blue = blue[sample_indices]
        original_indices = sample_indices
    else:
        original_indices = np.arange(total_points)
    
    num_points = len(x_coords)
    print(f"  Processing {num_points:,} points...")
    
    # Further reduce if still too many for memory constraints
    if num_points > 200000:
        print(f"  Warning: Large dataset may cause memory issues.")
        print(f"  Consider reducing --max-points or using --batch-size for batch processing.")
    
    # Convert RGB values to 0-255 range for easier color matching
    if has_rgb:
        # LAS files typically use 16-bit RGB (0-65535), convert to 8-bit (0-255)
        red_max_original = red.max() if red.max() > 0 else 65535
        green_max_original = green.max() if green.max() > 0 else 65535
        blue_max_original = blue.max() if blue.max() > 0 else 65535
        
        # Convert to 0-255 range
        red_255 = (red.astype(float) / red_max_original * 255.0).astype(np.uint8)
        green_255 = (green.astype(float) / green_max_original * 255.0).astype(np.uint8)
        blue_255 = (blue.astype(float) / blue_max_original * 255.0).astype(np.uint8)
        
        print(f"  Original RGB range: R={red.min()}-{red.max()}, G={green.min()}-{green.max()}, B={blue.min()}-{blue.max()}")
        print(f"  Converted to 0-255 range: R={red_255.min()}-{red_255.max()}, G={green_255.min()}-{green_255.max()}, B={blue_255.min()}-{blue_255.max()}")
        
        # Normalize to 0-1 for clustering (from 0-255 range)
        red_norm = red_255.astype(float) / 255.0
        green_norm = green_255.astype(float) / 255.0
        blue_norm = blue_255.astype(float) / 255.0
    else:
        print("  Error: No RGB color data found. Cannot cluster by color only.")
        raise ValueError("LAS file must contain RGB color data for color-based clustering.")
    
    # Create feature matrix: color only (normalized 0-1 from 0-255 range)
    print(f"  Creating color feature matrix (RGB 0-255 normalized to 0-1)...")
    
    # Use only RGB values for clustering
    features = np.column_stack([
        red_norm,
        green_norm,
        blue_norm
    ])
    
    # Choose clustering method
    print(f"  Clustering method: {method.upper()}")
    
    if SKLEARN_AVAILABLE:
        if method.lower() == 'kmeans':
            # K-means: Fast, fixed number of clusters - best for object extraction
            print(f"  Using K-means with {n_clusters} clusters (fast, fixed clusters)...")
            try:
                clusterer = KMeans(n_clusters=n_clusters, n_init=10, random_state=42, n_jobs=1)
                cluster_labels = clusterer.fit_predict(features)
                print(f"  ✓ K-means clustering complete!")
            except Exception as e:
                print(f"  K-means error: {e}, trying MiniBatchKMeans...")
                clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000, n_init=3)
                cluster_labels = clusterer.fit_predict(features)
                print(f"  ✓ MiniBatchKMeans clustering complete!")
        
        elif method.lower() == 'minibatch':
            # MiniBatch K-means: Even faster for very large datasets
            print(f"  Using MiniBatchKMeans with {n_clusters} clusters (very fast for large datasets)...")
            batch_size = min(1000, num_points // 10)
            clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=batch_size, n_init=3)
            cluster_labels = clusterer.fit_predict(features)
            print(f"  ✓ MiniBatchKMeans clustering complete!")
        
        elif method.lower() == 'dbscan':
            # DBSCAN: Variable number of clusters based on density
            eps_normalized = eps_color / 255.0
            print(f"  Using DBSCAN (color eps={eps_normalized:.4f}, min_samples={min_samples})...")
            n_jobs = 1 if num_points > 100000 else -1
            if n_jobs == 1:
                print(f"  Using single-threaded mode for memory efficiency...")
            
            try:
                clusterer = DBSCAN(eps=eps_normalized, min_samples=min_samples, metric='euclidean', n_jobs=n_jobs)
                cluster_labels = clusterer.fit_predict(features)
            except (MemoryError, OSError) as e:
                print(f"  Memory/system resource error: {e}")
                print(f"  Reducing dataset size and retrying...")
                if num_points > 50000:
                    reduce_to = 50000
                    print(f"  Reducing to {reduce_to:,} points for clustering...")
                    sample_indices = np.random.choice(num_points, reduce_to, replace=False)
                    features = features[sample_indices]
                    original_indices = original_indices[sample_indices]
                    x_coords = x_coords[sample_indices]
                    y_coords = y_coords[sample_indices]
                    z_coords = z_coords[sample_indices]
                    red = red[sample_indices]
                    green = green[sample_indices]
                    blue = blue[sample_indices]
                    num_points = reduce_to
                    
                    clusterer = DBSCAN(eps=eps_normalized, min_samples=min_samples, metric='euclidean', n_jobs=1)
                    cluster_labels = clusterer.fit_predict(features)
                else:
                    print(f"  Error: Dataset too large. Try reducing --max-points further.")
                    raise
        else:
            raise ValueError(f"Unknown method: {method}. Use 'kmeans', 'minibatch', or 'dbscan'")
    else:
        # Simple distance-based clustering (fallback)
        print("  Using simple distance-based clustering (install scikit-learn for better results)...")
        eps_normalized = eps_color / 255.0
        cluster_labels = simple_clustering(features, eps_normalized, min_samples)
    
    # Get unique cluster labels (excluding noise points labeled as -1 for DBSCAN)
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters >= 0]  # Remove noise (DBSCAN only)
    
    num_clusters = len(unique_clusters)
    noise_points = np.sum(cluster_labels == -1) if method.lower() == 'dbscan' else 0
    
    print(f"  Found {num_clusters} clusters")
    if noise_points > 0:
        print(f"  Noise points (not in any cluster): {noise_points:,}")
    
    # Set up output directory
    if output_dir is None:
        base_name = os.path.splitext(las_file_path)[0]
        output_dir = os.path.dirname(base_name) if os.path.dirname(base_name) else '.'
        base_name = os.path.basename(base_name)
    else:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(las_file_path))[0]
    
    output_files = []
    
    # Export each cluster
    print(f"\nExporting clusters...")
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_points = np.sum(cluster_mask)
        
        # Skip very small clusters (unless using K-means which creates all clusters)
        if method.lower() != 'kmeans' and method.lower() != 'minibatch' and cluster_points < min_samples:
            continue
        
        # Get original indices for this cluster
        cluster_original_indices = original_indices[cluster_mask]
        
        # Create output filename
        output_filename = f"{base_name}_cluster_{cluster_id:04d}_{cluster_points}pts.las"
        output_path = os.path.join(output_dir, output_filename)
        
        # Create new LAS file for this cluster
        out_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
        out_las.header.offsets = las.header.offsets
        out_las.header.scales = las.header.scales
        
        # Copy points in this cluster
        out_las.points = las.points[cluster_original_indices]
        
        # Update header with new bounds
        cluster_x = x_coords[cluster_mask]
        cluster_y = y_coords[cluster_mask]
        cluster_z = z_coords[cluster_mask]
        
        out_las.header.x_min = float(np.min(cluster_x))
        out_las.header.x_max = float(np.max(cluster_x))
        out_las.header.y_min = float(np.min(cluster_y))
        out_las.header.y_max = float(np.max(cluster_y))
        out_las.header.z_min = float(np.min(cluster_z))
        out_las.header.z_max = float(np.max(cluster_z))
        
        # Write to file
        out_las.write(output_path)
        output_files.append(output_path)
        
        # Calculate average color for this cluster
        if has_rgb:
            avg_red = np.mean(red[cluster_mask])
            avg_green = np.mean(green[cluster_mask])
            avg_blue = np.mean(blue[cluster_mask])
            print(f"  Cluster {cluster_id}: {cluster_points:,} points "
                  f"(RGB: {avg_red:.0f}, {avg_green:.0f}, {avg_blue:.0f}) -> {output_filename}")
        else:
            print(f"  Cluster {cluster_id}: {cluster_points:,} points -> {output_filename}")
    
    print(f"\n✓ Successfully created {len(output_files)} cluster files!")
    print(f"  Output directory: {os.path.abspath(output_dir)}")
    
    return output_files


def simple_clustering(features, eps, min_samples):
    """
    Simple distance-based clustering (fallback when scikit-learn is not available).
    """
    n_points = len(features)
    cluster_labels = np.full(n_points, -1)  # -1 means unassigned
    cluster_id = 0
    
    for i in range(n_points):
        if cluster_labels[i] != -1:
            continue
        
        # Find neighbors
        distances = np.sqrt(np.sum((features - features[i]) ** 2, axis=1))
        neighbors = np.where(distances <= eps)[0]
        
        if len(neighbors) >= min_samples:
            # Start new cluster
            cluster_labels[neighbors] = cluster_id
            cluster_id += 1
    
    return cluster_labels


def cluster_by_intensity(las_file_path, n_clusters=10, method='kmeans', eps_intensity=0.05,
                         min_samples=10, output_dir=None, max_points=200000):
    """
    Cluster points based on intensity similarity for object extraction.

    Args:
        las_file_path (str): Path to the input .las file
        n_clusters (int): Number of clusters to create (for K-means)
        method (str): Clustering method ('kmeans', 'minibatch', 'dbscan')
        eps_intensity (float): Maximum intensity distance for DBSCAN (0-1 scale)
        min_samples (int): Minimum points per cluster (for DBSCAN)
        output_dir (str): Directory to save output files (optional)
        max_points (int): Maximum points to process (for large files)

    Returns:
        list: List of output file paths
    """
    import laspy
    import numpy as np
    import os

    if not os.path.exists(las_file_path):
        raise FileNotFoundError(f"File not found: {las_file_path}")

    if not las_file_path.lower().endswith('.las'):
        raise ValueError(f"File must be a .las file: {las_file_path}")

    print(f"Reading LiDAR data from: {las_file_path}")
    las = laspy.read(las_file_path)

    # Coordinates
    x_coords = np.array(las.x)
    y_coords = np.array(las.y)
    z_coords = np.array(las.z)

    total_points = len(x_coords)
    print(f"  Total points: {total_points:,}")

    # Intensity
    if not hasattr(las, 'intensity'):
        raise ValueError("LAS file must contain intensity values for intensity-based clustering.")

    intensity_values = np.array(las.intensity, dtype=float)

    # Downsample if too many points
    if total_points > max_points:
        print(f"  Downsampling from {total_points:,} to {max_points:,} points for clustering...")
        sample_indices = np.random.choice(total_points, max_points, replace=False)
        x_coords = x_coords[sample_indices]
        y_coords = y_coords[sample_indices]
        z_coords = z_coords[sample_indices]
        intensity_values = intensity_values[sample_indices]
        original_indices = sample_indices
    else:
        original_indices = np.arange(total_points)

    num_points = len(x_coords)
    print(f"  Processing {num_points:,} points...")

    # Normalize intensity 0-1
    intensity_norm = (intensity_values / intensity_values.max()).reshape(-1, 1)

    # Clustering
    from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans

    print(f"  Clustering method: {method.upper()}")
    if method.lower() == 'kmeans':
        print(f"  Using K-means with {n_clusters} clusters...")
        clusterer = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = clusterer.fit_predict(intensity_norm)
    elif method.lower() == 'minibatch':
        batch_size = min(1000, num_points // 10)
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=batch_size, n_init=3)
        cluster_labels = clusterer.fit_predict(intensity_norm)
    elif method.lower() == 'dbscan':
        print(f"  Using DBSCAN (intensity eps={eps_intensity}, min_samples={min_samples})...")
        clusterer = DBSCAN(eps=eps_intensity, min_samples=min_samples, metric='euclidean', n_jobs=1)
        cluster_labels = clusterer.fit_predict(intensity_norm)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kmeans', 'minibatch', or 'dbscan'")

    # Unique clusters
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters >= 0]  # exclude noise (-1)
    num_clusters = len(unique_clusters)
    print(f"  Found {num_clusters} clusters")

    # Output directory
    if output_dir is None:
        base_name = os.path.splitext(las_file_path)[0]
        output_dir = os.path.dirname(base_name) or '.'
        base_name = os.path.basename(base_name)
    else:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(las_file_path))[0]

    output_files = []

    # Export clusters
    print(f"\nExporting clusters...")
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_points = np.sum(cluster_mask)
        cluster_original_indices = original_indices[cluster_mask]

        output_filename = f"{base_name}_intensity_cluster_{cluster_id:04d}_{cluster_points}pts.las"
        output_path = os.path.join(output_dir, output_filename)

        # New LAS file
        out_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
        out_las.header.offsets = las.header.offsets
        out_las.header.scales = las.header.scales
        out_las.points = las.points[cluster_original_indices]

        # Update header bounds
        out_las.header.x_min = float(np.min(x_coords[cluster_mask]))
        out_las.header.x_max = float(np.max(x_coords[cluster_mask]))
        out_las.header.y_min = float(np.min(y_coords[cluster_mask]))
        out_las.header.y_max = float(np.max(y_coords[cluster_mask]))
        out_las.header.z_min = float(np.min(z_coords[cluster_mask]))
        out_las.header.z_max = float(np.max(z_coords[cluster_mask]))

        # Write to file
        out_las.write(output_path)
        output_files.append(output_path)

        avg_intensity = np.mean(intensity_values[cluster_mask])
        print(
            f"  Cluster {cluster_id}: {cluster_points:,} points (Avg intensity: {avg_intensity:.2f}) -> {output_filename}")

    print(f"\n✓ Successfully created {len(output_files)} cluster files!")
    print(f"  Output directory: {os.path.abspath(output_dir)}")

    return output_files


def main():
    """Main function to run the script."""
    if len(sys.argv) < 2:
        print("Usage: python las_color_cluster.py <path_to_las_file> [OPTIONS]")
        print("\nOptions:")
        print("  --n-clusters VALUE      Number of clusters to create (default: 100)")
        print("  --method METHOD        Clustering method: 'kmeans' (fast, default), 'minibatch' (very fast), or 'dbscan' (variable clusters)")
        print("  --eps-color VALUE      Max color distance for DBSCAN (default: 30)")
        print("  --min-samples VALUE    Minimum points per cluster for DBSCAN (default: 10)")
        print("  --output-dir DIR       Output directory (default: same as input)")
        print("  --max-points VALUE     Max points to process (default: 200000)")
        print("\nExamples:")
        print("  # Fast K-means clustering with 100 clusters (recommended for object extraction)")
        print("  python las_color_cluster.py data.las")
        print("  # Custom number of clusters")
        print("  python las_color_cluster.py data.las --n-clusters 50")
        print("  # Very fast MiniBatch K-means for huge datasets")
        print("  python las_color_cluster.py data.las --method minibatch --n-clusters 100")
        print("  # DBSCAN (variable number of clusters)")
        print("  python las_color_cluster.py data.las --method dbscan --eps-color 20")
        sys.exit(1)
    
    las_file_path = sys.argv[1]
    
    # Parse arguments
    n_clusters = 2  # Default: good for object extraction
    method = 'kmeans'  # Default: fastest and best for fixed number of clusters
    eps_color = 55
    min_samples = 10
    nazwa_folderu = f"{sys.argv[1]}_Folder"
    if not os.path.exists(nazwa_folderu):
        os.mkdir(nazwa_folderu)
        print(f"Folder '{nazwa_folderu}' został utworzony.")
    else:
        nazwa_folderu = f"Brak_nazwy"
    output_dir = nazwa_folderu
    max_points = 10000000

    if '--n-clusters' in sys.argv:
        idx = sys.argv.index('--n-clusters')
        if idx + 1 < len(sys.argv):
            n_clusters = int(sys.argv[idx + 1])
    
    if '--method' in sys.argv:
        idx = sys.argv.index('--method')
        if idx + 1 < len(sys.argv):
            method = sys.argv[idx + 1].lower()
    
    if '--eps-color' in sys.argv:
        idx = sys.argv.index('--eps-color')
        if idx + 1 < len(sys.argv):
            eps_color = float(sys.argv[idx + 1])
    
    if '--min-samples' in sys.argv:
        idx = sys.argv.index('--min-samples')
        if idx + 1 < len(sys.argv):
            min_samples = int(sys.argv[idx + 1])
    
    if '--output-dir' in sys.argv:
        idx = sys.argv.index('--output-dir')
        if idx + 1 < len(sys.argv):
            output_dir = sys.argv[idx + 1]
    
    if '--max-points' in sys.argv:
        idx = sys.argv.index('--max-points')
        if idx + 1 < len(sys.argv):
            max_points = int(sys.argv[idx + 1])
    
    try:
        output_files = cluster_by_intensity(
            las_file_path,
            n_clusters=n_clusters,
            method=method,
            min_samples=min_samples,
            output_dir=output_dir,
            max_points=max_points
        )
        print(f"\n✓ All done! Created {len(output_files)} cluster files.")
        return output_files
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

