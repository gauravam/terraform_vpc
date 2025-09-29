import pandas as pd
import os
import json
from typing import List, Dict
import numpy as np


def format_size(size_bytes: int) -> str:
    """
    Format size in bytes to human-readable KB or MB.
    
    Parameters:
    -----------
    size_bytes : int
        Size in bytes
        
    Returns:
    --------
    str: Formatted size string (e.g., "4.68 KB" or "1.23 MB")
    """
    size_kb = size_bytes / 1024
    size_mb = size_bytes / (1024 * 1024)
    
    if size_mb >= 1:
        return f"{size_mb:.2f} MB"
    else:
        return f"{size_kb:.2f} KB"


def estimate_parquet_size(df: pd.DataFrame, compression_ratio: float = 0.3) -> int:
    """
    Estimate parquet file size based on dataframe metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to estimate size for
    compression_ratio : float
        Expected compression ratio (0.2-0.4 typical for parquet with compression)
        Lower = better compression. Default 0.3 means 30% of uncompressed size.
        
    Returns:
    --------
    int: Estimated size in bytes
    """
    # Get memory usage for each column
    memory_usage = df.memory_usage(deep=True, index=False)
    total_memory = memory_usage.sum()
    
    # Parquet metadata overhead (approximate)
    # Includes file header, column metadata, row group metadata, etc.
    base_overhead = 1024  # ~1KB base
    per_column_overhead = len(df.columns) * 100  # ~100 bytes per column
    per_row_group_overhead = 500  # ~500 bytes per row group
    
    # Estimate number of row groups (typically 1 for small partitions)
    num_row_groups = max(1, len(df) // 1000)
    
    # Calculate total overhead
    metadata_overhead = base_overhead + per_column_overhead + (per_row_group_overhead * num_row_groups)
    
    # Estimate compressed data size
    # Different data types compress differently
    compressed_size = 0
    
    for col in df.columns:
        col_memory = memory_usage[col]
        dtype = df[col].dtype
        
        # Adjust compression ratio based on data type
        if pd.api.types.is_integer_dtype(dtype):
            # Integers compress well, especially if sequential
            col_compression = compression_ratio * 0.8
        elif pd.api.types.is_float_dtype(dtype):
            # Floats compress moderately
            col_compression = compression_ratio * 1.0
        elif pd.api.types.is_string_dtype(dtype) or dtype == object:
            # Strings compress well if repetitive
            unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 1
            # More unique values = worse compression
            col_compression = compression_ratio * (0.5 + 0.5 * unique_ratio)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            # Dates compress very well
            col_compression = compression_ratio * 0.7
        elif pd.api.types.is_bool_dtype(dtype):
            # Booleans compress extremely well
            col_compression = compression_ratio * 0.5
        else:
            # Default compression
            col_compression = compression_ratio
        
        compressed_size += int(col_memory * col_compression)
    
    # Total estimated size
    estimated_size = compressed_size + metadata_overhead
    
    return max(estimated_size, metadata_overhead)  # At minimum, return overhead


def get_partition_stats(
    df: pd.DataFrame,
    partition_keys: List[str],
    base_path: str = "output",
    compression_ratio: float = 0.3,
    path_style: str = "hive"
) -> List[Dict[str, any]]:
    """
    Calculates Hive partition paths and statistics using memory-based metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to partition
    partition_keys : List[str]
        Column names to use as partition keys
    base_path : str
        Base directory path for partition paths (virtual path, not created)
    compression_ratio : float
        Expected compression ratio for parquet files (default 0.3 = 30% of original size)
        Range: 0.2 (aggressive) to 0.5 (conservative)
    path_style : str
        Path style for partition paths:
        - 'hive': Use forward slashes (default, works for S3 and most systems)
        - 's3': Ensure forward slashes for S3 compatibility
        - 'os': Use OS-specific separators (backslash on Windows)
        
    Returns:
    --------
    List[Dict[str, any]]
        List of dictionaries containing partition info:
        - partition_path: Hive-style partition path
        - num_rows: Number of rows in partition
        - size_bytes: Estimated size of parquet file in bytes
        - size_kb: Size in kilobytes
        - size_mb: Size in megabytes
        - size_formatted: Human-readable size string
        - partition_values: Dict of partition key-value pairs
        - method_used: 'estimated'
        
    Example:
    --------
    >>> df = pd.DataFrame({
    ...     'date': ['2024-01-01', '2024-01-01', '2024-01-02'],
    ...     'country': ['USA', 'UK', 'USA'],
    ...     'sales': [100, 200, 150]
    ... })
    >>> stats = get_partition_stats(df, ['date', 'country'])
    >>> for stat in stats:
    ...     print(f"{stat['partition_path']}: {stat['num_rows']} rows, {stat['size_formatted']}")
    """
    
    # Validate partition keys exist in dataframe
    for key in partition_keys:
        if key not in df.columns:
            raise ValueError(f"Partition key '{key}' not found in dataframe columns")
    
    # Calculate dataset memory usage
    dataset_memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Dataset size: {dataset_memory_mb:.2f} MB - Using metric-based estimation")
    
    partition_stats = []
    grouped = df.groupby(partition_keys)
    
    # Determine path separator based on style
    if path_style.lower() in ['hive', 's3']:
        path_sep = '/'
    elif path_style.lower() == 'os':
        path_sep = os.sep
    else:
        raise ValueError(f"Invalid path_style: {path_style}. Must be 'hive', 's3', or 'os'")
    
    # Calculate statistics for each partition
    for partition_values, group_df in grouped:
        if len(partition_keys) == 1:
            partition_values = (partition_values,)
        
        partition_parts = [
            f"{key}={value}" 
            for key, value in zip(partition_keys, partition_values)
        ]
        
        # Construct Hive-style partition path
        partition_path = path_sep.join([base_path] + partition_parts)
        
        # Estimate parquet size using metrics
        estimated_size = estimate_parquet_size(group_df, compression_ratio)
        
        partition_stats.append({
            'partition_path': partition_path,
            'num_rows': len(group_df),
            'size_bytes': estimated_size,
            'size_kb': round(estimated_size / 1024, 2),
            'size_mb': round(estimated_size / (1024 * 1024), 2),
            'size_formatted': format_size(estimated_size),
            'partition_values': dict(zip(partition_keys, partition_values)),
            'method_used': 'estimated'
        })
    
    return partition_stats


def print_partition_summary(stats: List[Dict]) -> None:
    """
    Pretty print partition statistics.
    
    Parameters:
    -----------
    stats : List[Dict]
        Output from get_partition_stats()
    """
    print("\n" + "="*100)
    print("PARTITION STATISTICS SUMMARY")
    print("="*100)
    
    total_rows = sum(s['num_rows'] for s in stats)
    total_size = sum(s['size_bytes'] for s in stats)
    
    print(f"\nTotal Partitions: {len(stats)}")
    print(f"Total Rows: {total_rows:,}")
    print(f"Total Size: {total_size:,} bytes ({format_size(total_size)})")
    print("\n" + "-"*100)
    
    for i, stat in enumerate(stats, 1):
        print(f"\nPartition {i}:")
        print(f"  Path: {stat['partition_path']}")
        print(f"  Values: {stat['partition_values']}")
        print(f"  Rows: {stat['num_rows']:,}")
        print(f"  Size: {stat['size_bytes']:,} bytes ({stat['size_formatted']})")
        print(f"  Method: {stat['method_used']}")
    
    print("\n" + "="*100)


def save_stats_to_json(stats: List[Dict], output_file: str, indent: int = 2) -> None:
    """
    Save partition statistics to a JSON file.
    
    Parameters:
    -----------
    stats : List[Dict]
        Output from get_partition_stats()
    output_file : str
        Path to output JSON file
    indent : int
        JSON indentation level (default: 2)
    """
    # Create summary statistics
    total_rows = sum(s['num_rows'] for s in stats)
    total_size = sum(s['size_bytes'] for s in stats)
    
    output_data = {
        'summary': {
            'total_partitions': len(stats),
            'total_rows': total_rows,
            'total_size_bytes': total_size,
            'total_size_kb': round(total_size / 1024, 2),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'total_size_formatted': format_size(total_size)
        },
        'partitions': stats
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=indent)
    
    print(f"\nStatistics saved to: {output_file}")


def load_stats_from_json(input_file: str) -> Dict:
    """
    Load partition statistics from a JSON file.
    
    Parameters:
    -----------
    input_file : str
        Path to input JSON file
        
    Returns:
    --------
    Dict containing 'summary' and 'partitions'
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Statistics loaded from: {input_file}")
    return data


# Example usage
if __name__ == "__main__":
    # Create sample dataset
    print("Creating sample dataset...")
    n_rows = 100
    sample_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n_rows, freq='h').strftime('%Y-%m-%d').tolist(),
        'country': (['USA', 'UK', 'Canada', 'Germany', 'France'] * (n_rows // 5 + 1))[:n_rows],
        'product': (['Product_A', 'Product_B', 'Product_C'] * (n_rows // 3 + 1))[:n_rows],
        'sales': list(range(1000, 1000 + n_rows)),
        'quantity': list(range(10, 10 + n_rows)),
        'revenue': [x * 10.5 for x in range(1000, 1000 + n_rows)]
    })
    
    print(f"Sample data shape: {sample_data.shape}")
    print(f"\nFirst few rows:")
    print(sample_data.head())
    
    # Example: Partition by multiple keys
    print("\n\n" + "="*100)
    print("EXAMPLE: Partitioning by 'date' and 'country' (Hive style)")
    print("="*100)
    stats = get_partition_stats(
        sample_data, 
        partition_keys=['date', 'country'], 
        base_path='data/partitioned',
        compression_ratio=0.3,  # 30% compression
        path_style='hive'
    )

    # Show first 5 partitions
    for i, stat in enumerate(stats[:5], 1):
        print(f"\nPartition {i}:")
        print(f"  Path: {stat['partition_path']}")
        print(f"  Values: {stat['partition_values']}")
        print(f"  Rows: {stat['num_rows']:,}")
        print(f"  Size: {stat['size_bytes']:,} bytes ({stat['size_formatted']})")
        print(f"  Method: {stat['method_used']}")
    
    print(f"\n... and {len(stats) - 5} more partitions")
    print("\n" + "="*100)
    
    # Print full summary
    print_partition_summary(stats)
    
    # Save to JSON
    save_stats_to_json(stats, 'partition_stats.json')
