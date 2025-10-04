"""
Data splitting module for SolarGuardAI project.
Implements temporal-aware train/validation/test splits for time series data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

# Set up logging
logger = logging.getLogger("SolarGuardAI-DataSplit")

class TemporalDataSplitter:
    """Handles temporal-aware data splitting for time series forecasting."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the temporal data splitter.
        
        Args:
            config: Configuration dictionary with splitting parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'test_size': 0.2,           # Proportion of data for testing
            'val_size': 0.2,            # Proportion of training data for validation
            'gap_days': 0,              # Gap days between train and test (to avoid leakage)
            'split_method': 'temporal', # 'temporal' or 'random'
            'random_seed': 42,          # Random seed for reproducibility
        }
        
        # Merge with provided config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
                
        logger.info("Initialized TemporalDataSplitter")
    
    def temporal_split(self, df: pd.DataFrame, time_col: str = 'beginTime') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally into train, validation, and test sets.
        
        Args:
            df: DataFrame with time series data
            time_col: Column name containing timestamps
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for temporal split")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
        if time_col not in df.columns:
            logger.error(f"Time column '{time_col}' not found in DataFrame")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
        try:
            # Make a copy to avoid modifying the original
            df_copy = df.copy()
            
            # Ensure time column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df_copy[time_col]):
                df_copy[time_col] = pd.to_datetime(df_copy[time_col])
            
            # Sort by time
            df_copy = df_copy.sort_values(time_col)
            
            # Calculate split indices
            total_size = len(df_copy)
            test_size = int(total_size * self.config['test_size'])
            
            # If gap days specified, implement gap between train and test
            if self.config['gap_days'] > 0:
                # Find the test start date
                test_start_idx = total_size - test_size
                test_start_date = df_copy.iloc[test_start_idx][time_col]
                
                # Calculate gap start date
                gap_start_date = test_start_date - timedelta(days=self.config['gap_days'])
                
                # Filter out gap period
                train_val_df = df_copy[df_copy[time_col] < gap_start_date]
                test_df = df_copy[df_copy[time_col] >= test_start_date]
            else:
                # No gap, simple split
                train_val_df = df_copy.iloc[:-test_size]
                test_df = df_copy.iloc[-test_size:]
            
            # Split train into train and validation
            train_size = int(len(train_val_df) * (1 - self.config['val_size']))
            train_df = train_val_df.iloc[:train_size]
            val_df = train_val_df.iloc[train_size:]
            
            logger.info(f"Temporal split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            logger.info(f"Train period: {train_df[time_col].min()} to {train_df[time_col].max()}")
            logger.info(f"Val period: {val_df[time_col].min()} to {val_df[time_col].max()}")
            logger.info(f"Test period: {test_df[time_col].min()} to {test_df[time_col].max()}")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            logger.error(f"Error in temporal split: {str(e)}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def random_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data randomly into train, validation, and test sets.
        Note: This should only be used for non-time-series data.
        
        Args:
            df: DataFrame to split
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for random split")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
        try:
            # Calculate effective validation size relative to total
            effective_val_size = self.config['val_size'] * (1 - self.config['test_size'])
            
            # First split into train_val and test
            train_val_df, test_df = train_test_split(
                df, 
                test_size=self.config['test_size'],
                random_state=self.config['random_seed']
            )
            
            # Then split train_val into train and validation
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=self.config['val_size'],
                random_state=self.config['random_seed']
            )
            
            logger.info(f"Random split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            logger.error(f"Error in random split: {str(e)}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def expanding_window_split(self, df: pd.DataFrame, time_col: str = 'beginTime', 
                              initial_train_size: float = 0.5, 
                              step_size: int = 30) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create expanding window splits for time series cross-validation.
        
        Args:
            df: DataFrame with time series data
            time_col: Column name containing timestamps
            initial_train_size: Initial proportion of data for first training window
            step_size: Number of days to expand window by in each step
            
        Returns:
            List of (train_df, test_df) tuples for each window
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for expanding window split")
            return []
            
        if time_col not in df.columns:
            logger.error(f"Time column '{time_col}' not found in DataFrame")
            return []
            
        try:
            # Make a copy to avoid modifying the original
            df_copy = df.copy()
            
            # Ensure time column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df_copy[time_col]):
                df_copy[time_col] = pd.to_datetime(df_copy[time_col])
            
            # Sort by time
            df_copy = df_copy.sort_values(time_col)
            
            # Calculate initial split
            total_size = len(df_copy)
            initial_train_idx = int(total_size * initial_train_size)
            
            # Get min and max dates
            min_date = df_copy[time_col].min()
            max_date = df_copy[time_col].max()
            
            # Calculate initial train end date
            initial_train_end_date = df_copy.iloc[initial_train_idx][time_col]
            
            # Create windows
            windows = []
            current_train_end = initial_train_end_date
            
            while current_train_end < max_date - timedelta(days=step_size):
                # Define test window end
                test_end = current_train_end + timedelta(days=step_size)
                
                # Create train and test sets
                train_df = df_copy[df_copy[time_col] <= current_train_end]
                test_df = df_copy[(df_copy[time_col] > current_train_end) & 
                                 (df_copy[time_col] <= test_end)]
                
                # Add to windows if test set is not empty
                if not test_df.empty:
                    windows.append((train_df, test_df))
                
                # Expand window
                current_train_end = test_end
            
            logger.info(f"Created {len(windows)} expanding windows for cross-validation")
            
            return windows
            
        except Exception as e:
            logger.error(f"Error in expanding window split: {str(e)}")
            return []
    
    def split_data(self, df: pd.DataFrame, time_col: str = 'beginTime') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data based on configured method.
        
        Args:
            df: DataFrame to split
            time_col: Column name containing timestamps (for temporal splits)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for splitting")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
        # Choose split method based on configuration
        if self.config['split_method'] == 'temporal':
            return self.temporal_split(df, time_col)
        else:
            logger.warning("Using random split for time series data is not recommended")
            return self.random_split(df)
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                   output_dir: str, prefix: str = 'solar_flare') -> bool:
        """
        Save train, validation, and test splits to CSV files.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            output_dir: Directory to save files
            prefix: Prefix for filenames
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import os
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save splits
            train_df.to_csv(f"{output_dir}/{prefix}_train.csv", index=False)
            val_df.to_csv(f"{output_dir}/{prefix}_val.csv", index=False)
            test_df.to_csv(f"{output_dir}/{prefix}_test.csv", index=False)
            
            logger.info(f"Saved data splits to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data splits: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example data splitting
    try:
        # Initialize splitter with custom config
        config = {
            'test_size': 0.2,
            'val_size': 0.15,
            'gap_days': 7,  # One week gap between train and test
            'split_method': 'temporal',
            'random_seed': 42
        }
        
        splitter = TemporalDataSplitter(config)
        
        # Load engineered data (example)
        # In a real scenario, this would come from the feature engineering step
        data_file = "path/to/engineered_flare_data.csv"
        
        # Check if file exists
        if os.path.exists(data_file):
            # Load data
            df = pd.read_csv(data_file, parse_dates=['beginTime'])
            
            # Split data
            train_df, val_df, test_df = splitter.split_data(df, time_col='beginTime')
            
            # Print results
            print(f"Train set: {len(train_df)} samples")
            print(f"Validation set: {len(val_df)} samples")
            print(f"Test set: {len(test_df)} samples")
            
            # Save splits
            output_dir = "data/processed"
            splitter.save_splits(train_df, val_df, test_df, output_dir)
            
        else:
            print(f"Engineered data file not found: {data_file}")
            print("Please run the feature engineering step first")
            
    except Exception as e:
        print(f"Error in example: {str(e)}")