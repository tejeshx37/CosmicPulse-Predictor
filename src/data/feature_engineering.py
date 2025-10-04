"""
Feature engineering module for SolarGuardAI project.
This module provides specialized feature engineering for solar flare prediction.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import scipy.stats as stats

# Set up logging
logger = logging.getLogger("SolarGuardAI-FeatureEngineering")

class SolarFlareFeatureEngineer:
    """Feature engineering for solar flare prediction models."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the feature engineer.
        
        Args:
            config: Configuration dictionary with feature engineering parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'use_solar_cycle_features': True,
            'use_temporal_features': True,
            'use_active_region_features': True,
            'use_flare_history_features': True,
            'lookback_windows': [24, 48, 72],  # Hours for historical aggregation
            'prediction_horizon': 24,  # Hours ahead to predict
        }
        
        # Merge with provided config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
                
        logger.info("Initialized SolarFlareFeatureEngineer")
    
    def add_solar_cycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add solar cycle-related features.
        
        Args:
            df: DataFrame with datetime information
            
        Returns:
            DataFrame with added solar cycle features
        """
        if not self.config['use_solar_cycle_features']:
            return df
            
        if df.empty or 'beginTime' not in df.columns:
            logger.warning("Cannot add solar cycle features: missing beginTime column")
            return df
            
        try:
            # Make a copy to avoid modifying the original
            df_solar = df.copy()
            
            # Extract basic time components if not already present
            if 'year' not in df_solar.columns:
                df_solar['year'] = df_solar['beginTime'].dt.year
            if 'month' not in df_solar.columns:
                df_solar['month'] = df_solar['beginTime'].dt.month
            if 'day_of_year' not in df_solar.columns:
                df_solar['day_of_year'] = df_solar['beginTime'].dt.dayofyear
            
            # Solar cycle position (approximate)
            # Solar Cycle 24 started in December 2008 and ended in December 2019
            # Solar Cycle 25 started in December 2019
            
            # Calculate years since start of cycle
            df_solar['solar_cycle_number'] = np.where(
                df_solar['beginTime'] >= pd.Timestamp('2019-12-01'),
                25,  # Current cycle (25)
                24   # Previous cycle (24)
            )
            
            # Calculate cycle start date
            df_solar['cycle_start_date'] = np.where(
                df_solar['solar_cycle_number'] == 25,
                pd.Timestamp('2019-12-01'),
                pd.Timestamp('2008-12-01')
            )
            
            # Calculate years into cycle
            df_solar['years_into_cycle'] = (
                (df_solar['beginTime'] - df_solar['cycle_start_date']).dt.days / 365.25
            )
            
            # Normalized position in cycle (0 to 1)
            df_solar['cycle_position'] = df_solar['years_into_cycle'] / 11
            
            # Sinusoidal features to capture cyclical nature
            df_solar['cycle_sin'] = np.sin(2 * np.pi * df_solar['cycle_position'])
            df_solar['cycle_cos'] = np.cos(2 * np.pi * df_solar['cycle_position'])
            
            # Drop intermediate columns
            df_solar = df_solar.drop(columns=['cycle_start_date'])
            
            logger.info("Added solar cycle features")
            return df_solar
            
        except Exception as e:
            logger.error(f"Error adding solar cycle features: {str(e)}")
            return df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features for time series analysis.
        
        Args:
            df: DataFrame with datetime information
            
        Returns:
            DataFrame with added temporal features
        """
        if not self.config['use_temporal_features']:
            return df
            
        if df.empty or 'beginTime' not in df.columns:
            logger.warning("Cannot add temporal features: missing beginTime column")
            return df
            
        try:
            # Make a copy to avoid modifying the original
            df_temporal = df.copy()
            
            # Extract time components if not already present
            if 'hour' not in df_temporal.columns:
                df_temporal['hour'] = df_temporal['beginTime'].dt.hour
            if 'day_of_week' not in df_temporal.columns:
                df_temporal['day_of_week'] = df_temporal['beginTime'].dt.dayofweek
            if 'day_of_year' not in df_temporal.columns:
                df_temporal['day_of_year'] = df_temporal['beginTime'].dt.dayofyear
            if 'month' not in df_temporal.columns:
                df_temporal['month'] = df_temporal['beginTime'].dt.month
            
            # Cyclical encoding of time features
            
            # Hour of day (0-23) -> sin/cos encoding
            df_temporal['hour_sin'] = np.sin(2 * np.pi * df_temporal['hour'] / 24)
            df_temporal['hour_cos'] = np.cos(2 * np.pi * df_temporal['hour'] / 24)
            
            # Day of week (0-6) -> sin/cos encoding
            df_temporal['day_of_week_sin'] = np.sin(2 * np.pi * df_temporal['day_of_week'] / 7)
            df_temporal['day_of_week_cos'] = np.cos(2 * np.pi * df_temporal['day_of_week'] / 7)
            
            # Day of year (1-366) -> sin/cos encoding
            df_temporal['day_of_year_sin'] = np.sin(2 * np.pi * df_temporal['day_of_year'] / 365.25)
            df_temporal['day_of_year_cos'] = np.cos(2 * np.pi * df_temporal['day_of_year'] / 365.25)
            
            # Month (1-12) -> sin/cos encoding
            df_temporal['month_sin'] = np.sin(2 * np.pi * df_temporal['month'] / 12)
            df_temporal['month_cos'] = np.cos(2 * np.pi * df_temporal['month'] / 12)
            
            # Time differences between events
            df_temporal['hours_since_previous'] = df_temporal['beginTime'].diff().dt.total_seconds() / 3600
            
            # Fill first row with median or reasonable value
            median_hours = df_temporal['hours_since_previous'].median()
            df_temporal.loc[0, 'hours_since_previous'] = median_hours if not pd.isna(median_hours) else 24
            
            # Add features for time since last significant flare (M or X class)
            if 'flare_class' in df_temporal.columns:
                # Create a mask for significant flares
                significant_mask = df_temporal['flare_class'].isin(['M', 'X'])
                
                # Get timestamps of significant flares
                significant_times = df_temporal.loc[significant_mask, 'beginTime'].reset_index(drop=True)
                
                # Initialize the new feature
                df_temporal['hours_since_significant'] = np.nan
                
                # For each flare, find the most recent significant flare
                for i, row in df_temporal.iterrows():
                    current_time = row['beginTime']
                    
                    # Find the most recent significant flare before this one
                    previous_significant = significant_times[significant_times < current_time]
                    
                    if not previous_significant.empty:
                        # Calculate hours since most recent significant flare
                        most_recent = previous_significant.iloc[-1]
                        hours_since = (current_time - most_recent).total_seconds() / 3600
                        df_temporal.loc[i, 'hours_since_significant'] = hours_since
                    else:
                        # No previous significant flare
                        df_temporal.loc[i, 'hours_since_significant'] = 720  # 30 days as default
            
            logger.info("Added temporal features")
            return df_temporal
            
        except Exception as e:
            logger.error(f"Error adding temporal features: {str(e)}")
            return df
    
    def add_active_region_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features related to active regions.
        
        Args:
            df: DataFrame with active region information
            
        Returns:
            DataFrame with added active region features
        """
        if not self.config['use_active_region_features']:
            return df
            
        if df.empty or 'activeRegionNum' not in df.columns:
            logger.warning("Cannot add active region features: missing activeRegionNum column")
            return df
            
        try:
            # Make a copy to avoid modifying the original
            df_ar = df.copy()
            
            # Ensure activeRegionNum is numeric
            df_ar['activeRegionNum'] = pd.to_numeric(df_ar['activeRegionNum'], errors='coerce')
            
            # Fill missing with 0 (indicating no active region)
            df_ar['activeRegionNum'] = df_ar['activeRegionNum'].fillna(0)
            
            # Create a flag for whether an active region is present
            df_ar['has_active_region'] = (df_ar['activeRegionNum'] > 0).astype(int)
            
            # For each active region, calculate its flare history
            if 'beginTime' in df_ar.columns and 'flare_intensity' in df_ar.columns:
                # Group by active region
                ar_groups = df_ar.groupby('activeRegionNum')
                
                # Initialize new columns
                df_ar['ar_flare_count'] = 0
                df_ar['ar_max_intensity'] = 0
                df_ar['ar_mean_intensity'] = 0
                df_ar['ar_days_active'] = 0
                
                # Calculate features for each active region
                for ar_num, group in ar_groups:
                    if ar_num == 0:  # Skip "no active region"
                        continue
                        
                    # Sort by time
                    group = group.sort_values('beginTime')
                    
                    # Calculate active region lifetime
                    ar_start = group['beginTime'].min()
                    ar_end = group['beginTime'].max()
                    ar_days = (ar_end - ar_start).total_seconds() / (24 * 3600)
                    
                    # Calculate flare history
                    ar_count = len(group)
                    ar_max = group['flare_intensity'].max()
                    ar_mean = group['flare_intensity'].mean()
                    
                    # Update values for this active region
                    ar_mask = df_ar['activeRegionNum'] == ar_num
                    df_ar.loc[ar_mask, 'ar_flare_count'] = ar_count
                    df_ar.loc[ar_mask, 'ar_max_intensity'] = ar_max
                    df_ar.loc[ar_mask, 'ar_mean_intensity'] = ar_mean
                    df_ar.loc[ar_mask, 'ar_days_active'] = ar_days
                    
                    # For each flare in this active region, calculate features based on prior flares
                    for i, row in group.iterrows():
                        current_time = row['beginTime']
                        
                        # Get prior flares in this active region
                        prior_flares = group[group['beginTime'] < current_time]
                        
                        if not prior_flares.empty:
                            # Count of prior flares
                            df_ar.loc[i, 'ar_prior_flare_count'] = len(prior_flares)
                            
                            # Max intensity of prior flares
                            df_ar.loc[i, 'ar_prior_max_intensity'] = prior_flares['flare_intensity'].max()
                            
                            # Mean intensity of prior flares
                            df_ar.loc[i, 'ar_prior_mean_intensity'] = prior_flares['flare_intensity'].mean()
                            
                            # Hours since first flare in this active region
                            first_flare = prior_flares['beginTime'].min()
                            hours_since_first = (current_time - first_flare).total_seconds() / 3600
                            df_ar.loc[i, 'ar_hours_since_first'] = hours_since_first
                            
                            # Hours since most recent flare in this active region
                            recent_flare = prior_flares['beginTime'].max()
                            hours_since_recent = (current_time - recent_flare).total_seconds() / 3600
                            df_ar.loc[i, 'ar_hours_since_recent'] = hours_since_recent
                        else:
                            # No prior flares
                            df_ar.loc[i, 'ar_prior_flare_count'] = 0
                            df_ar.loc[i, 'ar_prior_max_intensity'] = 0
                            df_ar.loc[i, 'ar_prior_mean_intensity'] = 0
                            df_ar.loc[i, 'ar_hours_since_first'] = 0
                            df_ar.loc[i, 'ar_hours_since_recent'] = 0
            
            logger.info("Added active region features")
            return df_ar
            
        except Exception as e:
            logger.error(f"Error adding active region features: {str(e)}")
            return df
    
    def add_flare_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features based on flare history across multiple time windows.
        
        Args:
            df: DataFrame with flare data
            
        Returns:
            DataFrame with added flare history features
        """
        if not self.config['use_flare_history_features']:
            return df
            
        if df.empty or 'beginTime' not in df.columns:
            logger.warning("Cannot add flare history features: missing beginTime column")
            return df
            
        try:
            # Make a copy to avoid modifying the original
            df_history = df.copy()
            
            # Sort by time
            df_history = df_history.sort_values('beginTime')
            
            # Features to aggregate
            agg_features = ['flare_intensity']
            if 'duration_minutes' in df_history.columns:
                agg_features.append('duration_minutes')
                
            # Get lookback windows from config
            windows = self.config['lookback_windows']
            
            # For each window, calculate aggregated features
            for window in windows:
                for feature in agg_features:
                    if feature not in df_history.columns:
                        continue
                        
                    # Initialize columns
                    df_history[f'{feature}_{window}h_count'] = 0
                    df_history[f'{feature}_{window}h_sum'] = 0
                    df_history[f'{feature}_{window}h_mean'] = 0
                    df_history[f'{feature}_{window}h_max'] = 0
                    df_history[f'{feature}_{window}h_std'] = 0
                    
                    # For each flare, calculate history in the window
                    for i, row in df_history.iterrows():
                        current_time = row['beginTime']
                        window_start = current_time - timedelta(hours=window)
                        
                        # Get flares in the window
                        window_mask = (
                            (df_history['beginTime'] >= window_start) & 
                            (df_history['beginTime'] < current_time)
                        )
                        window_flares = df_history.loc[window_mask]
                        
                        if not window_flares.empty:
                            # Calculate aggregates
                            df_history.loc[i, f'{feature}_{window}h_count'] = len(window_flares)
                            df_history.loc[i, f'{feature}_{window}h_sum'] = window_flares[feature].sum()
                            df_history.loc[i, f'{feature}_{window}h_mean'] = window_flares[feature].mean()
                            df_history.loc[i, f'{feature}_{window}h_max'] = window_flares[feature].max()
                            df_history.loc[i, f'{feature}_{window}h_std'] = window_flares[feature].std()
            
            # Add trend features
            if len(windows) > 1:
                windows.sort()  # Ensure windows are in ascending order
                
                for feature in agg_features:
                    if feature not in df_history.columns:
                        continue
                        
                    # Calculate trend between windows
                    for i in range(1, len(windows)):
                        w1 = windows[i-1]
                        w2 = windows[i]
                        
                        # Mean change between windows
                        df_history[f'{feature}_trend_{w1}h_to_{w2}h'] = (
                            df_history[f'{feature}_{w2}h_mean'] - df_history[f'{feature}_{w1}h_mean']
                        )
                        
                        # Count change between windows
                        df_history[f'{feature}_count_trend_{w1}h_to_{w2}h'] = (
                            df_history[f'{feature}_{w2}h_count'] - df_history[f'{feature}_{w1}h_count']
                        )
            
            logger.info(f"Added flare history features for {len(windows)} time windows")
            return df_history
            
        except Exception as e:
            logger.error(f"Error adding flare history features: {str(e)}")
            return df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for prediction at different horizons.
        
        Args:
            df: DataFrame with flare data
            
        Returns:
            DataFrame with added target variables
        """
        if df.empty or 'beginTime' not in df.columns:
            logger.warning("Cannot create target variables: missing beginTime column")
            return df
            
        try:
            # Make a copy to avoid modifying the original
            df_target = df.copy()
            
            # Sort by time
            df_target = df_target.sort_values('beginTime')
            
            # Get prediction horizon from config
            horizon = self.config['prediction_horizon']
            
            # Create target variables
            
            # 1. Will there be a flare in the next X hours?
            df_target[f'flare_occurs_{horizon}h'] = 0
            
            # 2. Will there be an M or X class flare in the next X hours?
            df_target[f'major_flare_{horizon}h'] = 0
            
            # 3. Maximum flare intensity in the next X hours
            df_target[f'max_intensity_{horizon}h'] = 0
            
            # For each flare, look ahead to see what happens in the prediction window
            for i, row in df_target.iterrows():
                current_time = row['beginTime']
                window_end = current_time + timedelta(hours=horizon)
                
                # Get flares in the future window
                future_mask = (
                    (df_target['beginTime'] > current_time) & 
                    (df_target['beginTime'] <= window_end)
                )
                future_flares = df_target.loc[future_mask]
                
                if not future_flares.empty:
                    # There is at least one flare in the window
                    df_target.loc[i, f'flare_occurs_{horizon}h'] = 1
                    
                    # Check for major flares (M or X class)
                    if 'flare_class' in future_flares.columns:
                        major_flares = future_flares[future_flares['flare_class'].isin(['M', 'X'])]
                        if not major_flares.empty:
                            df_target.loc[i, f'major_flare_{horizon}h'] = 1
                    
                    # Maximum intensity in the window
                    if 'flare_intensity' in future_flares.columns:
                        df_target.loc[i, f'max_intensity_{horizon}h'] = future_flares['flare_intensity'].max()
            
            logger.info(f"Created target variables for {horizon}h prediction horizon")
            return df_target
            
        except Exception as e:
            logger.error(f"Error creating target variables: {str(e)}")
            return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: DataFrame with preprocessed flare data
            
        Returns:
            DataFrame with engineered features
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for feature engineering")
            return df
            
        try:
            logger.info("Starting feature engineering pipeline")
            
            # Apply each feature engineering step
            df = self.add_solar_cycle_features(df)
            df = self.add_temporal_features(df)
            df = self.add_active_region_features(df)
            df = self.add_flare_history_features(df)
            df = self.create_target_variables(df)
            
            logger.info("Feature engineering pipeline completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering pipeline: {str(e)}")
            return df


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example feature engineering
    try:
        # Initialize feature engineer with custom config
        config = {
            'use_solar_cycle_features': True,
            'use_temporal_features': True,
            'use_active_region_features': True,
            'use_flare_history_features': True,
            'lookback_windows': [24, 48, 72, 168],  # 1, 2, 3, and 7 days
            'prediction_horizon': 24,  # 24 hours ahead
        }
        
        feature_engineer = SolarFlareFeatureEngineer(config)
        
        # Load preprocessed data (example)
        # In a real scenario, this would come from the preprocessor
        data_file = "path/to/preprocessed_flare_data.csv"
        
        # Check if file exists
        if os.path.exists(data_file):
            # Load data
            df = pd.read_csv(data_file, parse_dates=['beginTime'])
            
            # Apply feature engineering
            df_engineered = feature_engineer.engineer_features(df)
            
            # Print results
            print(f"Added {df_engineered.shape[1] - df.shape[1]} new features")
            print(f"Total features: {df_engineered.shape[1]}")
            
            # Save engineered data
            # df_engineered.to_csv("path/to/engineered_flare_data.csv", index=False)
        else:
            print(f"Preprocessed data file not found: {data_file}")
            print("Please run the preprocessing pipeline first")
            
    except Exception as e:
        print(f"Error in example: {str(e)}")