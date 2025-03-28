#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Value Counts Difference Module

This module provides functionality to compare value counts between DataFrame columns.
It handles different column names, data types, and provides options for normalization and sorting.
"""

import os
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional, Tuple


class ValueCountsDiffer:
    """
    A class to calculate and analyze the difference between value counts in DataFrames.
    Shows value counts of both columns and their difference.
    """
    
    def __init__(self, round_decimals: int = 2, normalize: bool = False, sort_index: bool = False):
        """
        Initialize the ValueCountsDiffer with configuration options.
        
        Parameters:
        -----------
        round_decimals : int, default=2
            Number of decimal places to round float values for comparison
        normalize : bool, default=False
            If True, return the relative frequencies instead of counts
        sort_index : bool, default=False
            If True, sort by index values instead of the difference
        """
        self.round_decimals = round_decimals
        self.normalize = normalize
        self.sort_index = sort_index
        
    def _validate_columns(self, 
                         df1: pd.DataFrame, 
                         df2: pd.DataFrame, 
                         cols1: Union[List[str], str, None], 
                         cols2: Union[List[str], str, None]) -> Tuple[List[str], List[str]]:
        """
        Validate and process column specifications.
        """
        # Process cols1
        if cols1 is None:
            cols1 = df1.columns.tolist()
        elif isinstance(cols1, str):
            cols1 = [cols1]
            
        # Process cols2
        if cols2 is None:
            if len(cols1) == 1:
                # If only one column in cols1, assume it's the same name in df2
                cols2 = cols1
            else:
                cols2 = df2.columns.tolist()
        elif isinstance(cols2, str):
            cols2 = [cols2]
        
        # Ensure equal number of columns to compare
        if len(cols1) != len(cols2):
            error_msg = f"Number of columns in cols1 ({len(cols1)}) and cols2 ({len(cols2)}) must be equal for comparison"
            raise ValueError(error_msg)
            
        return cols1, cols2
    
    def _prepare_series(self, 
                       series1: pd.Series, 
                       series2: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Prepare two series for comparison by handling data types and rounding if needed.
        """
        # Handle float values by rounding
        if series1.dtype == float:
            series1 = series1.round(self.round_decimals)
            
        if series2.dtype == float:
            series2 = series2.round(self.round_decimals)
            
        return series1, series2
    
    def _calculate_counts_and_diff(self, 
                                  series1: pd.Series, 
                                  series2: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Calculate the value counts of both series and their difference.
        
        Parameters:
        -----------
        series1 : pandas.Series
            First series to compare
        series2 : pandas.Series
            Second series to compare
            
        Returns:
        --------
        tuple
            DataFrame with value counts of both series and Series with their difference
        """
        # Get value counts
        counts1 = series1.value_counts(normalize=self.normalize)
        counts2 = series2.value_counts(normalize=self.normalize)
        
        # Reindex to ensure both have the same index values
        all_values = pd.Index(set(counts1.index).union(set(counts2.index)))
        counts1 = counts1.reindex(all_values, fill_value=0)
        counts2 = counts2.reindex(all_values, fill_value=0)
        
        # Combine into a DataFrame
        counts_df = pd.DataFrame({
            'first': counts1,
            'second': counts2,
            'difference': counts1 - counts2
        })
        
        # Sort as specified
        if self.sort_index:
            return counts_df.sort_index(ascending=False), counts_df['difference']
        else:
            return counts_df.sort_values('difference', ascending=False), counts_df['difference']
    
    def compare(self, 
               df1: pd.DataFrame, 
               df2: pd.DataFrame, 
               cols1: Union[List[str], str, None] = None, 
               cols2: Union[List[str], str, None] = None) -> Dict[str, Dict]:
        """
        Compare value counts between two DataFrames.
        
        Parameters:
        -----------
        df1 : pandas.DataFrame
            First DataFrame to compare
        df2 : pandas.DataFrame
            Second DataFrame to compare
        cols1 : str, list, or None, default=None
            Column name(s) from df1 to compare. If None, uses all columns in df1.
        cols2 : str, list, or None, default=None
            Column name(s) from df2 to compare. If None, uses same column names as cols1
            if cols1 is a single column, otherwise uses all columns in df2.
            
        Returns:
        --------
        dict
            Dictionary with keys as column comparisons and values as dictionary containing
            'counts' (DataFrame of value counts) and 'diff' (Series of differences)
        """
        print(f"Comparing DataFrames of shapes {df1.shape} and {df2.shape}")
        
        try:
            # Validate columns
            cols1, cols2 = self._validate_columns(df1, df2, cols1, cols2)
            
            # Dictionary to store results
            results = {}
            
            # Process each pair of columns
            for col1, col2 in zip(cols1, cols2):
                # Verify columns exist
                if col1 not in df1.columns:
                    print(f"Column '{col1}' not found in first DataFrame. Skipping.")
                    continue
                    
                if col2 not in df2.columns:
                    print(f"Column '{col2}' not found in second DataFrame. Skipping.")
                    continue
                
                print(f"Comparing columns: '{col1}' vs '{col2}'")
                
                try:
                    # Get series from each DataFrame
                    series1 = df1[col1]
                    series2 = df2[col2]
                    
                    # Prepare series (handle data types)
                    series1, series2 = self._prepare_series(series1, series2)
                    
                    # Calculate counts and difference
                    counts_df, diff = self._calculate_counts_and_diff(series1, series2)
                    
                    # Use a descriptive key for the result
                    result_key = f"{col1} vs {col2}" if col1 != col2 else col1
                    results[result_key] = {
                        'counts': counts_df,
                        'diff': diff
                    }
                    
                    print(f"Completed comparison for '{result_key}' with {len(diff)} unique values")
                    
                except Exception as e:
                    print(f"Error comparing columns '{col1}' and '{col2}': {str(e)}")
                    continue
            
            print(f"Comparison completed with {len(results)} successful column pairs")
            return results
            
        except Exception as e:
            print(f"Error during comparison: {str(e)}")
            raise


class ValueCountsReporter:
    """
    A class to format and output the results of value counts differences.
    """
    
    @staticmethod
    def print_summary(results_dict: Dict[str, Dict], 
                     top_n: Optional[int] = None) -> None:
        """
        Print a formatted summary of the counts and differences for each column.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary with column names as keys and dictionaries containing 'counts' and 'diff' as values
        top_n : int or None, default=None
            If specified, show only the top N differences (by absolute magnitude)
        """
        print(f"Printing summary for {len(results_dict)} column comparisons")
        
        for col, result in results_dict.items():
            counts_df = result['counts']
            
            print(f"\n=== {col} ===")
            print("Value\t\tFirst\t\tSecond\t\tDifference")
            print("-" * 60)
            
            # Optionally limit to top_n differences
            if top_n is not None:
                # Sort by absolute difference magnitude
                sorted_counts = counts_df.reindex(counts_df['difference'].abs().sort_values(ascending=False).index)
                counts_to_show = sorted_counts.head(top_n)
                print(f"Showing top {top_n} differences")
            else:
                counts_to_show = counts_df
            
            for value, row in counts_to_show.iterrows():
                # Format the output to align columns
                value_str = str(value)
                if len(value_str) < 8:
                    value_str = value_str + "\t"
                
                # Format the values (round if they're floats)
                first_val = f"{row['first']:.2f}" if isinstance(row['first'], float) else str(row['first'])
                second_val = f"{row['second']:.2f}" if isinstance(row['second'], float) else str(row['second'])
                diff_val = f"{row['difference']:.2f}" if isinstance(row['difference'], float) else str(row['difference'])
                
                print(f"{value_str}\t{first_val}\t\t{second_val}\t\t{diff_val}")
    
    @staticmethod
    def save_to_csv(results_dict: Dict[str, Dict], 
                   output_path: str) -> None:
        """
        Save the counts and difference results to a CSV file.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary with column names as keys and dictionaries containing 'counts' and 'diff' as values
        output_path : str
            Path where to save the CSV file
        """
        print(f"Saving counts and difference results to {output_path}")
        
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Create a multi-index DataFrame
            dfs = []
            for col, result in results_dict.items():
                df = result['counts'].copy()
                df.columns = pd.MultiIndex.from_product([[col], df.columns])
                dfs.append(df)
            
            if dfs:
                result_df = pd.concat(dfs, axis=1)
                # Save to CSV
                result_df.to_csv(output_path)
                print(f"Results saved successfully to {output_path}")
            else:
                print("No results to save")
            
        except Exception as e:
            print(f"Error saving results to CSV: {str(e)}")
            raise
    
    @staticmethod
    def save_to_excel(results_dict: Dict[str, Dict], 
                     output_path: str) -> None:
        """
        Save the counts and difference results to an Excel file with each comparison in a separate sheet.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary with column names as keys and dictionaries containing 'counts' and 'diff' as values
        output_path : str
            Path where to save the Excel file
        """
        print(f"Saving counts and difference results to Excel: {output_path}")
        
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Create Excel writer
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Create a summary sheet
                summary_data = []
                for col_name, result in results_dict.items():
                    counts_df = result['counts']
                    diff_series = result['diff']
                    
                    pos_diffs = sum(diff_series > 0)
                    neg_diffs = sum(diff_series < 0)
                    zero_diffs = sum(diff_series == 0)
                    max_diff = diff_series.max()
                    min_diff = diff_series.min()
                    
                    summary_data.append({
                        'Comparison': col_name,
                        'Total Values': len(diff_series),
                        'More in First': pos_diffs,
                        'Equal': zero_diffs,
                        'More in Second': neg_diffs,
                        'Max Difference': max_diff,
                        'Min Difference': min_diff
                    })
                
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Create individual sheets for each comparison
                for col_name, result in results_dict.items():
                    # Create a valid sheet name (Excel limits sheet names to 31 chars)
                    sheet_name = col_name[:31]
                    # Get the counts DataFrame
                    counts_df = result['counts']
                    # Sort by absolute difference
                    sorted_counts = counts_df.reindex(counts_df['difference'].abs().sort_values(ascending=False).index)
                    sorted_counts.to_excel(writer, sheet_name=sheet_name)
            
            print(f"Results saved successfully to Excel: {output_path}")
            
        except Exception as e:
            print(f"Error saving results to Excel: {str(e)}")
            raise


# Example usage
def main():
    """
    Example of using the ValueCountsDiffer and ValueCountsReporter classes.
    """
    # Sample data with different column names and float values
    df1 = pd.DataFrame({
        'category_a': ['A', 'B', 'A', 'C', 'A', 'B', 'D', 'E'],
        'price_a': [10.123, 15.456, 10.111, 20.789, 10.222, 15.333, 25.777, 30.888],
        'quantity_a': [5, 3, 5, 2, 5, 3, 1, 1]
    })
    
    df2 = pd.DataFrame({
        'category_b': ['A', 'C', 'C', 'D', 'B', 'F', 'F', 'F'],
        'price_b': [10.1, 20.7, 20.8, 25.8, 15.4, 35.9, 35.9, 30.9],
        'quantity_b': [4, 2, 2, 1, 3, 2, 2, 2]
    })
    
    try:
        # Create an instance with default settings
        differ = ValueCountsDiffer(round_decimals=2)
        
        # Compare columns with different names
        print("\nComparing category columns")
        diff_categories = differ.compare(df1, df2, 
                                        cols1=['category_a'], 
                                        cols2=['category_b'])
        
        # Print a summary
        ValueCountsReporter.print_summary(diff_categories)
        
        # Compare float columns with rounding
        print("\nComparing price columns")
        diff_prices = differ.compare(df1, df2, 
                                    cols1=['price_a'], 
                                    cols2=['price_b'])
        
        # Print a summary showing only top 3 differences
        ValueCountsReporter.print_summary(diff_prices, top_n=3)
        
        # Compare multiple columns at once with normalization
        print("\nComparing all columns with normalization")
        differ_normalized = ValueCountsDiffer(normalize=True)
        diff_all = differ_normalized.compare(df1, df2, 
                                           cols1=['category_a', 'price_a', 'quantity_a'], 
                                           cols2=['category_b', 'price_b', 'quantity_b'])
        
        # Create output directory
        os.makedirs("output", exist_ok=True)
        
        # Save the results to CSV and Excel
        ValueCountsReporter.save_to_csv(diff_all, "output/value_counts_comparison_results.csv")
        ValueCountsReporter.save_to_excel(diff_all, "output/value_counts_comparison_results.xlsx")
        
        print("\nExample completed successfully")
        
    except Exception as e:
        print(f"Error in example: {str(e)}")


if __name__ == "__main__":
    main()
