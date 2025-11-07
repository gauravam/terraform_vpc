import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class ColumnConfig:
    col1: str
    col2: str
    col_type: str  # 'J' (join), 'N' (numerical), 'C' (categorical)
    variance: Optional[float] = None


class TableReconciliation:
    
    def __init__(self, df1: pd.DataFrame, df2: pd.DataFrame, config: List[ColumnConfig], join_type: str = 'outer'):
        self.df1 = df1
        self.df2 = df2
        self.config = config
        self.merged = None
        self.results = {}
        self.join_type = join_type.lower()
        
        if self.join_type not in ['inner', 'outer']:
            raise ValueError("join_type must be 'inner' or 'outer'")
        
        join_cols = [c.col1 for c in config if c.col_type == 'J']
        if not join_cols:
            raise ValueError("At least one column marked as 'J' (join key) required")
        self.join_key = join_cols[0]
        
        self._run_all()
    
    def _run_all(self):
        """Auto-execute: join + compare all columns."""
        self.join()
        
        for cfg in self.config:
            if cfg.col_type == 'J':
                continue
            if cfg.col_type == 'N':
                self.compare_numerical(cfg.col1, cfg.col2, cfg.variance)
            elif cfg.col_type == 'C':
                self.compare_categorical(cfg.col1, cfg.col2)
    
    def join(self) -> Tuple[Optional[pd.DataFrame], Dict]:
        """Fast join with duplicate checking."""
        
        report = {
            't1_count': len(self.df1),
            't2_count': len(self.df2),
            'merged_count': None,
            'join_type': self.join_type,
            'status': 'FAILED',
            'errors': []
        }
        
        join_cfg = next(c for c in self.config if c.col_type == 'J')
        join_col_t1 = join_cfg.col1
        join_col_t2 = join_cfg.col2
        
        # Fast duplicate check using is_unique
        if not self.df1[join_col_t1].is_unique:
            dup_count = len(self.df1) - self.df1[join_col_t1].nunique()
            report['errors'].append(f"T1: {dup_count} duplicate {join_col_t1} keys")
        
        if not self.df2[join_col_t2].is_unique:
            dup_count = len(self.df2) - self.df2[join_col_t2].nunique()
            report['errors'].append(f"T2: {dup_count} duplicate {join_col_t2} keys")
        
        if report['errors']:
            return None, report
        
        # Fast merge
        try:
            df2_renamed = self.df2 if join_col_t2 == join_col_t1 else self.df2.rename(columns={join_col_t2: join_col_t1})
            
            self.merged = pd.merge(self.df1, df2_renamed, on=self.join_key, how=self.join_type,
                                   suffixes=('_t1', '_t2'), indicator=True)
            
            unmatched_t1 = (self.merged['_merge'] == 'left_only').sum()
            unmatched_t2 = (self.merged['_merge'] == 'right_only').sum()
            
            if unmatched_t1 > 0 or unmatched_t2 > 0:
                report['errors'].append(f"Unmatched: T1={unmatched_t1}, T2={unmatched_t2}")
            
            self.merged = self.merged.drop(columns=['_merge'])
            report['merged_count'] = len(self.merged)
            report['status'] = 'SUCCESS' if len(self.df1) == len(self.df2) == len(self.merged) else 'PARTIAL'
            
        except Exception as e:
            report['errors'].append(str(e))
            return None, report
        
        return self.merged, report
    
    def compare_numerical(self, col1: str, col2: str, variance: Optional[float] = None) -> Dict:
        """Fast vectorized numerical comparison using np.isclose."""
        
        if self.merged is None:
            return {'error': 'Run join() first'}
        
        c1 = col1 if col1 in self.merged.columns else f"{col1}_t1"
        c2 = col2 if col2 in self.merged.columns else f"{col2}_t2"
        
        # Vectorized conversion
        v1 = pd.to_numeric(self.merged[c1], errors='coerce').values
        v2 = pd.to_numeric(self.merged[c2], errors='coerce').values
        
        diff = v1 - v2
        
        # Fast vectorized comparison
        matches = np.zeros(len(self.merged), dtype=bool)
        try:
            if variance is not None:
                matches = np.isclose(v1, v2, rtol=variance, atol=0, equal_nan=True)
            else:
                matches = np.isclose(v1, v2, rtol=0, atol=0, equal_nan=True)
            # Both NaN matches
            both_nan = np.isnan(v1) & np.isnan(v2)
            matches = matches | both_nan
        except Exception:
            pass
        
        matching = matches.sum()
        total = len(self.merged)
        
        result = {
            'col1': col1, 'col2': col2, 'type': 'NUMERICAL',
            'total': total, 'matching': matching, 'mismatching': total - matching,
            'match_pct': (matching / total * 100) if total > 0 else 0,
            'variance': variance,
            'stats': {
                'mean_diff': np.nanmean(diff),
                'max_diff': np.nanmax(np.abs(diff)),
                'std_dev': np.nanstd(diff)
            },
            'matches': matches,
            'diff': diff
        }
        
        self.results[col1] = result
        return result
    
    def compare_categorical(self, col1: str, col2: str) -> Dict:
        """Fast vectorized string comparison."""
        
        if self.merged is None:
            return {'error': 'Run join() first'}
        
        c1 = col1 if col1 in self.merged.columns else f"{col1}_t1"
        c2 = col2 if col2 in self.merged.columns else f"{col2}_t2"
        
        # Vectorized string operations
        v1 = self.merged[c1].astype(str).str.strip().values
        v2 = self.merged[c2].astype(str).str.strip().values
        
        matches = v1 == v2
        matching = matches.sum()
        total = len(self.merged)
        
        result = {
            'col1': col1, 'col2': col2, 'type': 'CATEGORICAL',
            'total': total, 'matching': matching, 'mismatching': total - matching,
            'match_pct': (matching / total * 100) if total > 0 else 0,
            'matches': matches
        }
        
        self.results[col1] = result
        return result
    
    def analyze(self, col1: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fast analysis with minimal copying."""
        
        if self.merged is None:
            return pd.DataFrame(), pd.DataFrame()
        
        cfg = next((c for c in self.config if c.col1 == col1), None)
        if not cfg:
            return pd.DataFrame(), pd.DataFrame()
        
        col2 = cfg.col2
        col_type = cfg.col_type
        
        if col1 not in self.results:
            if col_type == 'N':
                self.compare_numerical(col1, col2, cfg.variance)
            else:
                self.compare_categorical(col1, col2)
        
        matches = self.results[col1]['matches']
        
        c1 = col1 if col1 in self.merged.columns else f"{col1}_t1"
        c2 = col2 if col2 in self.merged.columns else f"{col2}_t2"
        
        # Use view instead of copy where possible
        matched = self.merged.loc[matches, [self.join_key, c1, c2]].copy()
        matched.columns = [self.join_key, col1, col2]
        
        mismatched = self.merged.loc[~matches, [self.join_key, c1, c2]].copy()
        mismatched.columns = [self.join_key, col1, col2]
        
        if col_type == 'N' and len(mismatched) > 0:
            v1 = pd.to_numeric(self.merged.loc[~matches, c1], errors='coerce').values
            v2 = pd.to_numeric(self.merged.loc[~matches, c2], errors='coerce').values
            diff = v1 - v2
            mismatched['diff'] = diff
            mismatched = mismatched.iloc[np.argsort(np.abs(diff))[::-1]]
        elif len(mismatched) > 0:
            mismatched['diff'] = mismatched[col1].astype(str) + ' → ' + mismatched[col2].astype(str)
        
        return matched, mismatched
    
    def summary(self) -> pd.DataFrame:
        """Fast summary generation."""
        data = [
            {
                'Column (T1)': result['col1'],
                'Column (T2)': result['col2'],
                'Type': result['type'],
                'Total': result['total'],
                'Match': result['matching'],
                'Mismatch': result['mismatching'],
                'Match %': f"{result['match_pct']:.2f}%"
            }
            for result in self.results.values()
        ]
        return pd.DataFrame(data)
