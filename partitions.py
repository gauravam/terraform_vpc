import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO
import os
import tempfile
import shutil
import json
from pathlib import Path
from typing import List, Dict


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


def get_partition_stats(
    df: pd.DataFrame,
    partition_keys: List[str],
    base_path: str = "output",
    memory_threshold_mb: float = 500,
    path_style: str = "hive"
) -> List[Dict[str, any]]:
    """
    Calculates Hive partition paths and statistics with automatic mode selection.
    
    Automatically chooses the best method based on dataset size:
    - Small datasets (<threshold): Write to memory for accuracy
    - Large datasets (>=threshold): Use temp files to avoid memory errors
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to partition
    partition_keys : List[str]
        Column names to use as partition keys
    base_path : str
        Base directory path for partition paths (virtual path, not created)
    memory_threshold_mb : float
        Memory size threshold in MB. If dataset exceeds this, use temp files.
        Default is 500 MB.
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
        - size_bytes: Size of parquet file in bytes
        - size_kb: Size in kilobytes
        - size_mb: Size in megabytes
        - size_formatted: Human-readable size string
        - partition_values: Dict of partition key-value pairs
        - method_used: 'memory', 'temp_files', or 'estimated'
        
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
    
    # Automatically choose method based on size
    use_temp_files = dataset_memory_mb >= memory_threshold_mb
    
    if use_temp_files:
        print(f"Dataset size: {dataset_memory_mb:.2f} MB - Using temp files for safety")
    else:
        print(f"Dataset size: {dataset_memory_mb:.2f} MB - Using memory buffers")
    
    partition_stats = []
    grouped = df.groupby(partition_keys, dropna=False)
    
    # Determine path separator based on style
    if path_style.lower() in ['hive', 's3']:
        path_sep = '/'
    elif path_style.lower() == 'os':
        path_sep = os.sep
    else:
        raise ValueError(f"Invalid path_style: {path_style}. Must be 'hive', 's3', or 'os'")
    
    if use_temp_files:
        # Use temporary directory for large datasets
        temp_dir = tempfile.mkdtemp()
        try:
            table = pa.Table.from_pandas(df)
            pq.write_to_dataset(
                table,
                root_path=temp_dir,
                partition_cols=partition_keys,
                existing_data_behavior='overwrite_or_ignore'
            )
            
            for partition_values, group_df in grouped:
                if len(partition_keys) == 1:
                    partition_values = (partition_values,)
                
                partition_parts = [
                    f"{key}={value}" 
                    for key, value in zip(partition_keys, partition_values)
                ]
                
                # Construct Hive-style partition path
                partition_path = path_sep.join([base_path] + partition_parts)
                
                # Local temp path always uses OS separator
                temp_partition_path = os.path.join(temp_dir, *partition_parts)
                
                # Get actual file size from temp directory
                partition_dir = Path(temp_partition_path)
                parquet_files = list(partition_dir.glob("*.parquet"))
                file_size = parquet_files[0].stat().st_size if parquet_files else 0
                
                partition_stats.append({
                    'partition_path': partition_path,
                    'num_rows': len(group_df),
                    'size_bytes': file_size,
                    'size_kb': round(file_size / 1024, 2),
                    'size_mb': round(file_size / (1024 * 1024), 2),
                    'size_formatted': format_size(file_size),
                    'partition_values': dict(zip(partition_keys, partition_values)),
                    'method_used': 'temp_files'
                })
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    else:
        # Write to memory for smaller datasets
        for partition_values, group_df in grouped:
            if len(partition_keys) == 1:
                partition_values = (partition_values,)
            
            partition_parts = [
                f"{key}={value}" 
                for key, value in zip(partition_keys, partition_values)
            ]
            
            # Construct Hive-style partition path
            partition_path = path_sep.join([base_path] + partition_parts)
            
            try:
                # Calculate parquet size in memory
                buffer = BytesIO()
                table = pa.Table.from_pandas(group_df)
                pq.write_table(table, buffer, compression='snappy')
                file_size = buffer.tell()
                buffer.close()
                
                partition_stats.append({
                    'partition_path': partition_path,
                    'num_rows': len(group_df),
                    'size_bytes': file_size,
                    'size_kb': round(file_size / 1024, 2),
                    'size_mb': round(file_size / (1024 * 1024), 2),
                    'size_formatted': format_size(file_size),
                    'partition_values': dict(zip(partition_keys, partition_values)),
                    'method_used': 'memory'
                })
            except MemoryError:
                # Fallback to estimation if memory error occurs
                print(f"Memory error on partition {partition_values}. Using estimation.")
                estimated_size = int(group_df.memory_usage(deep=True).sum() * 0.3)
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














import json
from IPython.display import HTML
import statistics

def create_partition_dashboard(json_data):
    """
    Creates an HTML dashboard from partition statistics JSON data.
    
    Args:
        json_data: Either a JSON string or a dict containing partition statistics
    """
    # Parse JSON if string
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # Extract metrics
    total_partitions = data['summary']['total_partitions']
    partitions = data['partitions']
    
    # Calculate statistics
    partition_sizes_mb = [p['size_mb'] for p in partitions]
    partition_sizes_kb = [p['size_kb'] for p in partitions]
    partition_rows = [p['num_rows'] for p in partitions]
    
    avg_size_mb = sum(partition_sizes_mb) / len(partition_sizes_mb)
    median_size_mb = statistics.median(partition_sizes_mb)
    avg_size_kb = sum(partition_sizes_kb) / len(partition_sizes_kb)
    median_size_kb = statistics.median(partition_sizes_kb)
    
    # Format size intelligently
    if avg_size_mb < 1:
        avg_size_display = f"{avg_size_kb:.2f} KB"
        median_size_display = f"{median_size_kb:.2f} KB"
    else:
        avg_size_display = f"{avg_size_mb:.2f} MB"
        median_size_display = f"{median_size_mb:.2f} MB"
    
    avg_rows = sum(partition_rows) / len(partition_rows)
    median_rows = statistics.median(partition_rows)
    
    # Calculate skew (coefficient of variation for rows)
    std_rows = statistics.stdev(partition_rows)
    cv_rows = (std_rows / avg_rows) * 100 if avg_rows > 0 else 0
    max_rows = max(partition_rows)
    min_rows = min(partition_rows)
    range_rows = max_rows - min_rows
    
    # Determine skew color (red if CV > 30%)
    skew_color = '#e74c3c' if cv_rows > 30 else '#27ae60'
    
    # Count partitions by size
    under_1mb = sum(1 for s in partition_sizes_mb if s < 1)
    between_1_25mb = sum(1 for s in partition_sizes_mb if 1 <= s < 25)
    between_25_128mb = sum(1 for s in partition_sizes_mb if 25 <= s < 128)
    over_400mb = sum(1 for s in partition_sizes_mb if s >= 400)
    
    # Create binned histogram for better visualization with many partitions
    num_bins = min(100, total_partitions)  # Cap at 100 bins for performance
    if total_partitions > num_bins:
        # Create bins
        bin_size = len(partition_rows) // num_bins
        binned_rows = []
        for i in range(0, len(partition_rows), bin_size):
            bin_data = partition_rows[i:i+bin_size]
            if bin_data:
                binned_rows.append(sum(bin_data) / len(bin_data))
        histogram_data = binned_rows
    else:
        histogram_data = partition_rows
    
    max_hist_value = max(histogram_data)
    
    # Create HTML
    html = f"""
    <style>
        .dashboard {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding: 20px;
            background: #f5f7fa;
            border-radius: 8px;
        }}
        .metrics-row {{
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-box {{
            flex: 1;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 8px;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-subvalue {{
            font-size: 14px;
            color: #95a5a6;
            margin-top: 4px;
        }}
        .skew-section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .section-title {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
        }}
        .skew-bar {{
            height: 60px;
            background: linear-gradient(to right, {skew_color}, {skew_color});
            border-radius: 4px;
            position: relative;
            margin: 20px 0;
        }}
        .skew-info {{
            display: flex;
            justify-content: space-between;
            font-size: 14px;
            color: #7f8c8d;
        }}
        .size-boxes {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .size-box {{
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 6px solid;
            position: relative;
            overflow: hidden;
            transition: transform 0.2s ease;
        }}
        .size-box:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        .size-box.red {{
            background: linear-gradient(135deg, #ffeaea 0%, #ffcdd2 100%);
            border-left-color: #c62828;
        }}
        .size-box.orange {{
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
            border-left-color: #e65100;
        }}
        .size-box.green {{
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            border-left-color: #2e7d32;
        }}
        .size-box.yellow {{
            background: linear-gradient(135deg, #fff3e0 0%, #ffccbc 100%);
            border-left-color: #d84315;
        }}
        .size-header {{
            font-size: 13px;
            font-weight: 600;
            color: #37474f;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .size-count {{
            font-size: 42px;
            font-weight: bold;
            margin: 8px 0;
        }}
        .size-box.red .size-count {{
            color: #b71c1c;
        }}
        .size-box.orange .size-count {{
            color: #e65100;
        }}
        .size-box.green .size-count {{
            color: #1b5e20;
        }}
        .size-box.yellow .size-count {{
            color: #bf360c;
        }}
        .size-label {{
            font-size: 13px;
            color: #546e7a;
            margin-top: 4px;
            font-weight: 600;
        }}
        .histogram {{
            display: flex;
            align-items: flex-end;
            height: 120px;
            gap: 1px;
            margin: 15px 0;
            background: #ecf0f1;
            padding: 5px;
            border-radius: 4px;
            overflow-x: auto;
        }}
        .hist-bar {{
            flex: 1;
            background: {skew_color};
            border-radius: 2px 2px 0 0;
            min-height: 2px;
            min-width: 2px;
            transition: opacity 0.2s;
        }}
        .hist-bar:hover {{
            opacity: 0.7;
        }}
        .skew-stats {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 15px;
        }}
        .skew-stat {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 6px;
            text-align: center;
        }}
        .skew-stat-label {{
            font-size: 12px;
            color: #7f8c8d;
            margin-bottom: 4px;
        }}
        .skew-stat-value {{
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
        }}
    </style>
    
    <div class="dashboard">
        <div class="metrics-row">
            <div class="metric-box">
                <div class="metric-label">Total Partitions</div>
                <div class="metric-value">{total_partitions}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Partition Size</div>
                <div class="metric-value">{avg_size_display}</div>
                <div class="metric-subvalue">avg · median: {median_size_display}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Rows per Partition</div>
                <div class="metric-value">{avg_rows:.1f}</div>
                <div class="metric-subvalue">avg · median: {median_rows:.0f}</div>
            </div>
        </div>
        
        <div class="skew-section">
            <div class="section-title">Data Skew Analysis</div>
            <div class="skew-info">
                <span>Partitions: {total_partitions}</span>
                <span>Coefficient of Variation: {cv_rows:.1f}%</span>
                <span>Range: {range_rows} rows</span>
            </div>
            <div class="histogram">
                {''.join([f'<div class="hist-bar" style="height: {(r/max_hist_value)*100}%;" title="{r:.1f} rows"></div>' for r in histogram_data])}
            </div>
            <div class="skew-stats">
                <div class="skew-stat">
                    <div class="skew-stat-label">Minimum</div>
                    <div class="skew-stat-value">{min_rows}</div>
                </div>
                <div class="skew-stat">
                    <div class="skew-stat-label">Average</div>
                    <div class="skew-stat-value">{avg_rows:.1f}</div>
                </div>
                <div class="skew-stat">
                    <div class="skew-stat-label">Maximum</div>
                    <div class="skew-stat-value">{max_rows}</div>
                </div>
            </div>
            <div class="skew-info" style="margin-top: 15px; justify-content: center;">
                <span style="color: {skew_color}; font-weight: bold; font-size: 15px;">
                    {'⚠ High Skew Detected' if cv_rows > 30 else '✓ Low Skew'}
                </span>
            </div>
        </div>
        
        <div class="section-title">Partition Size Distribution</div>
        <div class="size-boxes">
            <div class="size-box red">
                <div class="size-header">&lt; 1 MB</div>
                <div class="size-count">{under_1mb}</div>
                <div class="size-label">partitions · Too Small</div>
            </div>
            <div class="size-box orange">
                <div class="size-header">1 - 25 MB</div>
                <div class="size-count">{between_1_25mb}</div>
                <div class="size-label">partitions · Good</div>
            </div>
            <div class="size-box green">
                <div class="size-header">25 - 128 MB</div>
                <div class="size-count">{between_25_128mb}</div>
                <div class="size-label">partitions · Optimal</div>
            </div>
            <div class="size-box yellow">
                <div class="size-header">&gt; 400 MB</div>
                <div class="size-count">{over_400mb}</div>
                <div class="size-label">partitions · ⚠ Caution</div>
            </div>
        </div>
    </div>
    """
    
    return HTML(html)

# Example usage:
# with open('partition_stats.json', 'r') as f:
#     json_data = json.load(f)
# create_partition_dashboard(json_data)
