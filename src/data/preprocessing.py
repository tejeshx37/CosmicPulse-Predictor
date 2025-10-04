"""
Data preprocessing pipeline for SolarGuardAI project.
This module handles cleaning, normalization, and preparation of solar flare data.
"""

import os
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any

# Set up logging
logger = logging.getLogger("SolarGuardAI-Preprocessing")

class SolarFlarePreprocessor:
    """Preprocessor for solar flare data from NASA DONKI API."""
    
    # Flare class mapping for numerical representation
    FLARE_CLASS_MAPPING = {
        'A': 1, 'B': 2, 'C': 3, 'M': 4, 'X': 5
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the solar flare preprocessor.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'min_flare_class': 'C',  # Minimum flare class to include
            'time_window_hours': 24,  # Time window for feature aggregation
            'fill_missing_strategy': 'interpolate',  # Strategy for missing values
            'outlier_threshold': 3.0,  # Standard deviations for outlier detection
            'include_derived_features': True,  # Whether to include derived features
        }
        
        # Merge with provided config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
                
        logger.info("Initialized SolarFlarePreprocessor")
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load solar flare data from JSON file.
        
        Args:
            file_path: Path to the JSON file containing flare data
            
        Returns:
            DataFrame with loaded flare data
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return pd.DataFrame()
                
            # Load JSON data
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
                logger.info(f"Loaded {len(df)} records from {file_path}")
                return df
            else:
                logger.error(f"Invalid data format in {file_path}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the solar flare data.
        
        Args:
            df: DataFrame with raw flare data
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for cleaning")
            return df
            
        try:
            # Make a copy to avoid modifying the original
            df_clean = df.copy()
            
            # Convert timestamps to datetime
            for col in ['beginTime', 'peakTime', 'endTime']:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            
            # Filter out records with missing critical data
            required_cols = ['flrID', 'beginTime', 'classType']
            for col in required_cols:
                if col in df_clean.columns:
                    df_clean = df_clean.dropna(subset=[col])
            
            # Filter by minimum flare class if specified
            if 'classType' in df_clean.columns and self.config['min_flare_class'] in self.FLARE_CLASS_MAPPING:
                min_class_value = self.FLARE_CLASS_MAPPING[self.config['min_flare_class']]
                
                # Extract flare class letter and convert to numerical value
                df_clean['flare_class_letter'] = df_clean['classType'].str[0]
                df_clean['flare_class_value'] = df_clean['flare_class_letter'].map(self.FLARE_CLASS_MAPPING)
                
                # Filter by minimum class
                df_clean = df_clean[df_clean['flare_class_value'] >= min_class_value]
            
            # Sort by begin time
            if 'beginTime' in df_clean.columns:
                df_clean = df_clean.sort_values('beginTime')
            
            # Reset index
            df_clean = df_clean.reset_index(drop=True)
            
            logger.info(f"Cleaned data: {len(df_clean)} records remaining")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            return df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and engineer features from solar flare data.
        
        Args:
            df: DataFrame with cleaned flare data
            
        Returns:
            DataFrame with extracted features
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for feature extraction")
            return df
            
        try:
            # Make a copy to avoid modifying the original
            df_features = df.copy()
            
            # Extract flare class and magnitude
            if 'classType' in df_features.columns:
                # Extract class letter (e.g., X, M, C)
                df_features['flare_class'] = df_features['classType'].str[0]
                
                # Extract magnitude (e.g., 1.2 from X1.2)
                df_features['flare_magnitude'] = df_features['classType'].str[1:].astype(float)
                
                # Create numerical representation of flare intensity
                # For example, X1.0 = 5.1, M5.0 = 4.5, C3.0 = 3.3
                df_features['flare_intensity'] = df_features.apply(
                    lambda row: self.FLARE_CLASS_MAPPING.get(row['flare_class'], 0) + 
                                (row['flare_magnitude'] / 10.0),
                    axis=1
                )
            
            # Calculate flare duration in minutes
            if all(col in df_features.columns for col in ['beginTime', 'endTime']):
                df_features['duration_minutes'] = (
                    (df_features['endTime'] - df_features['beginTime']).dt.total_seconds() / 60
                )
                
                # Handle negative or missing durations
                df_features.loc[df_features['duration_minutes'] < 0, 'duration_minutes'] = np.nan
                
                # Fill missing durations with median
                median_duration = df_features['duration_minutes'].median()
                df_features['duration_minutes'] = df_features['duration_minutes'].fillna(median_duration)
            
            # Extract time-based features
            if 'beginTime' in df_features.columns:
                df_features['year'] = df_features['beginTime'].dt.year
                df_features['month'] = df_features['beginTime'].dt.month
                df_features['day'] = df_features['beginTime'].dt.day
                df_features['hour'] = df_features['beginTime'].dt.hour
                df_features['day_of_year'] = df_features['beginTime'].dt.dayofyear
                
                # Solar cycle position (approximate)
                # Assuming a standard 11-year solar cycle
                df_features['solar_cycle_pos'] = ((df_features['year'] % 11) + 
                                                 df_features['day_of_year'] / 365) / 11
            
            # Extract active region features if available
            if 'activeRegionNum' in df_features.columns:
                # Convert to numeric, handling missing values
                df_features['activeRegionNum'] = pd.to_numeric(
                    df_features['activeRegionNum'], errors='coerce'
                )
                
                # Fill missing with 0 (indicating no active region)
                df_features['activeRegionNum'] = df_features['activeRegionNum'].fillna(0)
            
            # Add derived features if configured
            if self.config['include_derived_features']:
                # Add time since previous flare (in hours)
                if 'beginTime' in df_features.columns:
                    df_features['hours_since_previous'] = df_features['beginTime'].diff().dt.total_seconds() / 3600
                    df_features.loc[0, 'hours_since_previous'] = 24  # Default for first record
                
                # Add flare intensity change from previous
                if 'flare_intensity' in df_features.columns:
                    df_features['intensity_change'] = df_features['flare_intensity'].diff()
                    df_features.loc[0, 'intensity_change'] = 0  # Default for first record
            
            logger.info(f"Extracted features from {len(df_features)} records")
            return df_features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with handled missing values
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for missing value handling")
            return df
            
        try:
            # Make a copy to avoid modifying the original
            df_filled = df.copy()
            
            # Get numeric columns
            numeric_cols = df_filled.select_dtypes(include=['number']).columns
            
            # Apply strategy based on configuration
            strategy = self.config['fill_missing_strategy']
            
            if strategy == 'interpolate':
                # Interpolate numeric columns
                for col in numeric_cols:
                    if df_filled[col].isna().any():
                        df_filled[col] = df_filled[col].interpolate(method='linear')
                        
                        # Fill remaining NAs (at edges) with forward/backward fill
                        df_filled[col] = df_filled[col].fillna(method='ffill').fillna(method='bfill')
                        
            elif strategy == 'mean':
                # Fill with mean for each numeric column
                for col in numeric_cols:
                    if df_filled[col].isna().any():
                        mean_val = df_filled[col].mean()
                        df_filled[col] = df_filled[col].fillna(mean_val)
                        
            elif strategy == 'median':
                # Fill with median for each numeric column
                for col in numeric_cols:
                    if df_filled[col].isna().any():
                        median_val = df_filled[col].median()
                        df_filled[col] = df_filled[col].fillna(median_val)
                        
            elif strategy == 'zero':
                # Fill with zeros for each numeric column
                for col in numeric_cols:
                    if df_filled[col].isna().any():
                        df_filled[col] = df_filled[col].fillna(0)
            
            # For categorical columns, fill with most frequent value
            cat_cols = df_filled.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if df_filled[col].isna().any():
                    mode_val = df_filled[col].mode()[0]
                    df_filled[col] = df_filled[col].fillna(mode_val)
            
            # For datetime columns, forward fill
            date_cols = df_filled.select_dtypes(include=['datetime']).columns
            for col in date_cols:
                if df_filled[col].isna().any():
                    df_filled[col] = df_filled[col].fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"Handled missing values using strategy: {strategy}")
            return df_filled
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            return df
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers in the dataset.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with handled outliers
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for outlier handling")
            return df
            
        try:
            # Make a copy to avoid modifying the original
            df_no_outliers = df.copy()
            
            # Get numeric columns
            numeric_cols = df_no_outliers.select_dtypes(include=['number']).columns
            
            # Skip columns that should not be processed for outliers
            skip_cols = ['year', 'month', 'day', 'hour', 'flare_class_value']
            numeric_cols = [col for col in numeric_cols if col not in skip_cols]
            
            # Apply outlier detection and handling
            threshold = self.config['outlier_threshold']
            
            for col in numeric_cols:
                # Calculate mean and standard deviation
                mean_val = df_no_outliers[col].mean()
                std_val = df_no_outliers[col].std()
                
                # Identify outliers (values beyond threshold standard deviations)
                lower_bound = mean_val - threshold * std_val
                upper_bound = mean_val + threshold * std_val
                
                # Replace outliers with bounds
                df_no_outliers.loc[df_no_outliers[col] < lower_bound, col] = lower_bound
                df_no_outliers.loc[df_no_outliers[col] > upper_bound, col] = upper_bound
            
            logger.info(f"Handled outliers using threshold: {threshold} standard deviations")
            return df_no_outliers
            
        except Exception as e:
            logger.error(f"Error handling outliers: {str(e)}")
            return df
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numeric features to a standard scale.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with normalized features
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for normalization")
            return df
            
        try:
            # Make a copy to avoid modifying the original
            df_norm = df.copy()
            
            # Get numeric columns
            numeric_cols = df_norm.select_dtypes(include=['number']).columns
            
            # Skip columns that should not be normalized
            skip_cols = ['year', 'month', 'day', 'hour', 'flare_class_value']
            numeric_cols = [col for col in numeric_cols if col not in skip_cols]
            
            # Apply min-max normalization
            for col in numeric_cols:
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                
                # Avoid division by zero
                if max_val > min_val:
                    df_norm[f'{col}_norm'] = (df_norm[col] - min_val) / (max_val - min_val)
                else:
                    df_norm[f'{col}_norm'] = 0
            
            logger.info(f"Normalized {len(numeric_cols)} numeric features")
            return df_norm
            
        except Exception as e:
            logger.error(f"Error normalizing features: {str(e)}")
            return df
    
    def create_time_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-windowed features for time series analysis.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with time-windowed features
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for time window creation")
            return df
            
        if 'beginTime' not in df.columns:
            logger.error("beginTime column required for time window creation")
            return df
            
        try:
            # Make a copy to avoid modifying the original
            df_windows = df.copy()
            
            # Sort by time to ensure correct window creation
            df_windows = df_windows.sort_values('beginTime')
            
            # Features to aggregate
            agg_features = ['flare_intensity', 'duration_minutes']
            agg_features = [f for f in agg_features if f in df_windows.columns]
            
            if not agg_features:
                logger.warning("No features available for time window aggregation")
                return df_windows
            
            # Time window in hours
            window_hours = self.config['time_window_hours']
            
            # Create aggregated features
            for feature in agg_features:
                # Initialize new columns
                df_windows[f'{feature}_24h_count'] = 0
                df_windows[f'{feature}_24h_sum'] = 0
                df_windows[f'{feature}_24h_max'] = 0
                df_windows[f'{feature}_24h_mean'] = 0
                
                # For each row, calculate aggregates for previous window
                for i in range(len(df_windows)):
                    # Current time
                    current_time = df_windows.iloc[i]['beginTime']
                    
                    # Window start time
                    window_start = current_time - timedelta(hours=window_hours)
                    
                    # Get events in the window
                    window_mask = (df_windows['beginTime'] >= window_start) & (df_windows['beginTime'] < current_time)
                    window_events = df_windows.loc[window_mask]
                    
                    # Calculate aggregates
                    if not window_events.empty:
                        df_windows.loc[df_windows.index[i], f'{feature}_24h_count'] = len(window_events)
                        df_windows.loc[df_windows.index[i], f'{feature}_24h_sum'] = window_events[feature].sum()
                        df_windows.loc[df_windows.index[i], f'{feature}_24h_max'] = window_events[feature].max()
                        df_windows.loc[df_windows.index[i], f'{feature}_24h_mean'] = window_events[feature].mean()
            
            logger.info(f"Created time-windowed features with {window_hours}h window")
            return df_windows
            
        except Exception as e:
            logger.error(f"Error creating time windows: {str(e)}")
            return df
    
    def prepare_for_training(self, df: pd.DataFrame, target_col: str = 'flare_intensity') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare the final dataset for model training.
        
        Args:
            df: DataFrame with processed features
            target_col: Column to use as prediction target
            
        Returns:
            Tuple of (features_df, target_series)
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for training preparation")
            return pd.DataFrame(), pd.Series()
            
        if target_col not in df.columns:
            logger.error(f"Target column {target_col} not found in DataFrame")
            return pd.DataFrame(), pd.Series()
            
        try:
            # Make a copy to avoid modifying the original
            df_final = df.copy()
            
            # Extract target variable
            y = df_final[target_col]
            
            # Remove columns not useful for training
            drop_cols = [
                'flrID', 'beginTime', 'peakTime', 'endTime', 'sourceLocation',
                'activeRegionNum', 'linkedEvents', 'link', 'classType'
            ]
            
            # Only drop columns that exist
            drop_cols = [col for col in drop_cols if col in df_final.columns]
            
            # Drop target column from features
            drop_cols.append(target_col)
            
            # Create feature DataFrame
            X = df_final.drop(columns=drop_cols, errors='ignore')
            
            # Convert categorical columns to dummy variables
            cat_cols = X.select_dtypes(include=['object']).columns
            if not cat_cols.empty:
                X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            
            logger.info(f"Prepared dataset with {X.shape[1]} features and {len(y)} samples")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing for training: {str(e)}")
            return pd.DataFrame(), pd.Series()
    
    def process_pipeline(self, file_path: str, target_col: str = 'flare_intensity') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            file_path: Path to the raw data file
            target_col: Column to use as prediction target
            
        Returns:
            Tuple of (features_df, target_series)
        """
        try:
            logger.info(f"Starting preprocessing pipeline for {file_path}")
            
            # Load data
            df = self.load_data(file_path)
            if df.empty:
                return pd.DataFrame(), pd.Series()
            
            # Clean data
            df = self.clean_data(df)
            
            # Extract features
            df = self.extract_features(df)
            
            # Handle missing values
            df = self.handle_missing_values(df)
            
            # Handle outliers
            df = self.handle_outliers(df)
            
            # Create time windows
            df = self.create_time_windows(df)
            
            # Normalize features
            df = self.normalize_features(df)
            
            # Prepare for training
            X, y = self.prepare_for_training(df, target_col)
            
            logger.info("Preprocessing pipeline completed successfully")
            return X, y
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            return pd.DataFrame(), pd.Series()


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example preprocessing
    try:
        # Initialize preprocessor with custom config
        config = {
            'min_flare_class': 'C',
            'time_window_hours': 48,
            'fill_missing_strategy': 'interpolate',
            'outlier_threshold': 2.5,
            'include_derived_features': True
        }
        
        preprocessor = SolarFlarePreprocessor(config)
        
        # Process sample data
        data_file = "path/to/solar_flare_data.json"
        
        # Check if file exists
        if os.path.exists(data_file):
            # Run pipeline
            X, y = preprocessor.process_pipeline(data_file)
            
            # Print results
            if not X.empty:
                print(f"Processed {len(X)} samples with {X.shape[1]} features")
                print(f"Feature columns: {X.columns.tolist()}")
            else:
                print("Preprocessing failed or no data available")
        else:
            print(f"Sample data file not found: {data_file}")
            print("Please run the NASA DONKI API data collection first")
            
    except Exception as e:
        print(f"Error in example: {str(e)}")