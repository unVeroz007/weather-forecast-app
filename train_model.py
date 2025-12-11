<<<<<<< HEAD
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weather Temperature Forecasting - Model Training Script
This script trains a Random Forest model for temperature prediction
and saves it for deployment.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

warnings.filterwarnings('ignore')

def generate_synthetic_data(n_samples=10000):
    """
    Generate synthetic weather data for demonstration.
    In production, replace with real data loading.
    """
    print("📊 Generating synthetic weather data...")
    
    np.random.seed(42)
    
    # Generate timestamps
    start_date = datetime(2020, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_samples, freq='H')
    
    # Generate realistic seasonal patterns
    hour_of_day = np.array([d.hour for d in dates])
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    
    # Base patterns
    seasonal_pattern = 10 * np.sin(2 * np.pi * day_of_year / 365)
    diurnal_pattern = 5 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi/2)
    
    # Generate features with realistic relationships
    base_temp = 15 + seasonal_pattern + diurnal_pattern
    
    # Temperature with noise
    temperature = base_temp + np.random.normal(0, 2, n_samples)
    
    # Humidity inversely related to temperature
    humidity = 70 - 0.5 * (temperature - 15) + np.random.normal(0, 5, n_samples)
    humidity = np.clip(humidity, 20, 100)
    
    # Pressure with seasonal variation
    pressure = 1013 + 8 * np.sin(2 * np.pi * day_of_year / 180) + np.random.normal(0, 3, n_samples)
    
    # Wind speed with some correlation to pressure gradient
    wind_speed = 10 + 0.1 * np.abs(pressure - 1013) + np.random.exponential(2, n_samples)
    wind_speed = np.clip(wind_speed, 0, 40)
    
    # Visibility inversely related to humidity
    visibility = 15 - 0.1 * humidity + np.random.normal(0, 2, n_samples)
    visibility = np.clip(visibility, 0, 20)
    
    # Apparent temperature (wind chill/heat index simplified)
    apparent_temp = temperature - 0.1 * wind_speed * (1 - humidity/100) + np.random.normal(0, 1, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'Temperature (C)': temperature,
        'Humidity': humidity,
        'Pressure (millibars)': pressure,
        'Wind Speed (km/h)': wind_speed,
        'Visibility (km)': visibility,
        'Apparent Temperature (C)': apparent_temp
    })
    
    print(f"✅ Generated {len(df)} samples")
    print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

def create_lag_features(df, lag_hours=24):
    """
    Create lag features for time series prediction.
    """
    print(f"🔄 Creating lag features ({lag_hours} hours)...")
    
    # Features to create lags for
    lag_cols = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 
                'Visibility (km)', 'Pressure (millibars)']
    
    df_lag = df.copy()
    
    for col in lag_cols:
        for lag in range(1, lag_hours + 1):
            df_lag[f'{col}_lag{lag}'] = df_lag[col].shift(lag)
    
    # Also create some time-based features
    df_lag['hour'] = df_lag['timestamp'].dt.hour
    df_lag['month'] = df_lag['timestamp'].dt.month
    df_lag['day_of_week'] = df_lag['timestamp'].dt.dayofweek
    
    # Cyclical encoding for hour and month
    df_lag['hour_sin'] = np.sin(2 * np.pi * df_lag['hour'] / 24)
    df_lag['hour_cos'] = np.cos(2 * np.pi * df_lag['hour'] / 24)
    df_lag['month_sin'] = np.sin(2 * np.pi * df_lag['month'] / 12)
    df_lag['month_cos'] = np.cos(2 * np.pi * df_lag['month'] / 12)
    
    # Drop original timestamp and cyclical raw columns
    df_lag = df_lag.drop(columns=['timestamp', 'hour', 'month', 'day_of_week'])
    
    # Drop rows with NaN from lagging
    initial_len = len(df_lag)
    df_lag = df_lag.dropna()
    final_len = len(df_lag)
    
    print(f"✅ Created {len(lag_cols) * lag_hours + 4} lag features")
    print(f"📉 Dropped {initial_len - final_len} rows with NaN")
    
    return df_lag

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train Random Forest model with optimized hyperparameters.
    """
    print("🌲 Training Random Forest model...")
    
    # Reduced n_estimators for faster training in deployment
    model = RandomForestRegressor(
        n_estimators=200,           # Reduced from 300 for faster inference
        max_depth=20,               # Limit depth to prevent overfitting
        min_samples_split=5,        # Increased for regularization
        min_samples_leaf=2,         # Increased for regularization
        max_features='sqrt',        # Use sqrt features for diversity
        bootstrap=True,
        random_state=42,
        n_jobs=-1,                  # Use all cores
        verbose=0
    )
    
    # Train model
    import time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✅ Model trained in {training_time:.2f} seconds")
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train': {
            'mae': mean_absolute_error(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'r2': r2_score(y_train, y_pred_train)
        },
        'test': {
            'mae': mean_absolute_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'r2': r2_score(y_test, y_pred_test)
        }
    }
    
    # Print metrics
    print("\n📊 Model Performance:")
    print("=" * 50)
    print(f"{'Metric':<15} {'Training':<15} {'Test':<15}")
    print("-" * 50)
    print(f"{'MAE (°C)':<15} {metrics['train']['mae']:<15.3f} {metrics['test']['mae']:<15.3f}")
    print(f"{'RMSE (°C)':<15} {metrics['train']['rmse']:<15.3f} {metrics['test']['rmse']:<15.3f}")
    print(f"{'R² Score':<15} {metrics['train']['r2']:<15.3f} {metrics['test']['r2']:<15.3f}")
    print("=" * 50)
    
    return model, metrics

def save_model_and_scaler(model, scaler, metrics):
    """
    Save the trained model and scaler to disk.
    """
    print("💾 Saving model and scaler...")
    
    # Save model with compression
    with open('model_rf.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save metrics for reference
    with open('model_metrics.json', 'w') as f:
        import json
        json.dump(metrics, f, indent=2)
    
    # Print model info
    print(f"✅ Model saved: {os.path.getsize('model_rf.pkl') / 1024 / 1024:.2f} MB")
    print(f"✅ Scaler saved: {os.path.getsize('scaler.pkl') / 1024:.2f} KB")
    
    # Print feature importance summary
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Get top 10 features
        indices = np.argsort(importances)[::-1][:10]
        
        print("\n🏆 Top 10 Feature Importances:")
        print("-" * 40)
        for i, idx in enumerate(indices):
            print(f"{i+1:2d}. Feature {idx:3d}: {importances[idx]:.4f}")
        print("-" * 40)

def main():
    """Main training pipeline."""
    print("=" * 60)
    print("🌡️ WEATHER TEMPERATURE FORECASTING - MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Generate/Load data
    df = generate_synthetic_data(n_samples=5000)  # Reduced for faster training
    
    # Step 2: Create lag features
    df_lag = create_lag_features(df, lag_hours=24)
    
    # Step 3: Prepare features and target
    X = df_lag.drop(columns=['Temperature (C)'])
    y = df_lag['Temperature (C)']
    
    print(f"\n📐 Dataset shape: {X.shape}")
    print(f"🎯 Target shape: {y.shape}")
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # No shuffle for time series
    )
    
    print(f"\n📊 Data Split:")
    print(f"  Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Step 5: Scale features
    print("\n⚖️ Scaling features...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features scaled")
    
    # Step 6: Train model
    model, metrics = train_random_forest(
        X_train_scaled, y_train, 
        X_test_scaled, y_test
    )
    
    # Step 7: Save model
    save_model_and_scaler(model, scaler, metrics)
    
    # Step 8: Generate sample predictions
    print("\n🔍 Sample Predictions:")
    print("-" * 40)
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    for i, idx in enumerate(sample_indices):
        actual = y_test.iloc[idx]
        predicted = model.predict([X_test_scaled[idx]])[0]
        error = predicted - actual
        print(f"Sample {i+1}: Actual={actual:.2f}°C, "
              f"Predicted={predicted:.2f}°C, "
              f"Error={error:+.2f}°C")
    
    print("\n" + "=" * 60)
    print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n📁 Files generated:")
    print("  - model_rf.pkl (trained model)")
    print("  - scaler.pkl (fitted scaler)")
    print("  - model_metrics.json (performance metrics)")
    
    print("\n🚀 Next steps:")
    print("  1. Run the app: streamlit run app.py")
    print("  2. Deploy to Streamlit Cloud")
    print("  3. Test predictions in the web interface")

if __name__ == "__main__":
=======
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weather Temperature Forecasting - Model Training Script
This script trains a Random Forest model for temperature prediction
and saves it for deployment.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

warnings.filterwarnings('ignore')

def generate_synthetic_data(n_samples=10000):
    """
    Generate synthetic weather data for demonstration.
    In production, replace with real data loading.
    """
    print("📊 Generating synthetic weather data...")
    
    np.random.seed(42)
    
    # Generate timestamps
    start_date = datetime(2020, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_samples, freq='H')
    
    # Generate realistic seasonal patterns
    hour_of_day = np.array([d.hour for d in dates])
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    
    # Base patterns
    seasonal_pattern = 10 * np.sin(2 * np.pi * day_of_year / 365)
    diurnal_pattern = 5 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi/2)
    
    # Generate features with realistic relationships
    base_temp = 15 + seasonal_pattern + diurnal_pattern
    
    # Temperature with noise
    temperature = base_temp + np.random.normal(0, 2, n_samples)
    
    # Humidity inversely related to temperature
    humidity = 70 - 0.5 * (temperature - 15) + np.random.normal(0, 5, n_samples)
    humidity = np.clip(humidity, 20, 100)
    
    # Pressure with seasonal variation
    pressure = 1013 + 8 * np.sin(2 * np.pi * day_of_year / 180) + np.random.normal(0, 3, n_samples)
    
    # Wind speed with some correlation to pressure gradient
    wind_speed = 10 + 0.1 * np.abs(pressure - 1013) + np.random.exponential(2, n_samples)
    wind_speed = np.clip(wind_speed, 0, 40)
    
    # Visibility inversely related to humidity
    visibility = 15 - 0.1 * humidity + np.random.normal(0, 2, n_samples)
    visibility = np.clip(visibility, 0, 20)
    
    # Apparent temperature (wind chill/heat index simplified)
    apparent_temp = temperature - 0.1 * wind_speed * (1 - humidity/100) + np.random.normal(0, 1, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'Temperature (C)': temperature,
        'Humidity': humidity,
        'Pressure (millibars)': pressure,
        'Wind Speed (km/h)': wind_speed,
        'Visibility (km)': visibility,
        'Apparent Temperature (C)': apparent_temp
    })
    
    print(f"✅ Generated {len(df)} samples")
    print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

def create_lag_features(df, lag_hours=24):
    """
    Create lag features for time series prediction.
    """
    print(f"🔄 Creating lag features ({lag_hours} hours)...")
    
    # Features to create lags for
    lag_cols = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 
                'Visibility (km)', 'Pressure (millibars)']
    
    df_lag = df.copy()
    
    for col in lag_cols:
        for lag in range(1, lag_hours + 1):
            df_lag[f'{col}_lag{lag}'] = df_lag[col].shift(lag)
    
    # Also create some time-based features
    df_lag['hour'] = df_lag['timestamp'].dt.hour
    df_lag['month'] = df_lag['timestamp'].dt.month
    df_lag['day_of_week'] = df_lag['timestamp'].dt.dayofweek
    
    # Cyclical encoding for hour and month
    df_lag['hour_sin'] = np.sin(2 * np.pi * df_lag['hour'] / 24)
    df_lag['hour_cos'] = np.cos(2 * np.pi * df_lag['hour'] / 24)
    df_lag['month_sin'] = np.sin(2 * np.pi * df_lag['month'] / 12)
    df_lag['month_cos'] = np.cos(2 * np.pi * df_lag['month'] / 12)
    
    # Drop original timestamp and cyclical raw columns
    df_lag = df_lag.drop(columns=['timestamp', 'hour', 'month', 'day_of_week'])
    
    # Drop rows with NaN from lagging
    initial_len = len(df_lag)
    df_lag = df_lag.dropna()
    final_len = len(df_lag)
    
    print(f"✅ Created {len(lag_cols) * lag_hours + 4} lag features")
    print(f"📉 Dropped {initial_len - final_len} rows with NaN")
    
    return df_lag

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train Random Forest model with optimized hyperparameters.
    """
    print("🌲 Training Random Forest model...")
    
    # Reduced n_estimators for faster training in deployment
    model = RandomForestRegressor(
        n_estimators=200,           # Reduced from 300 for faster inference
        max_depth=20,               # Limit depth to prevent overfitting
        min_samples_split=5,        # Increased for regularization
        min_samples_leaf=2,         # Increased for regularization
        max_features='sqrt',        # Use sqrt features for diversity
        bootstrap=True,
        random_state=42,
        n_jobs=-1,                  # Use all cores
        verbose=0
    )
    
    # Train model
    import time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✅ Model trained in {training_time:.2f} seconds")
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train': {
            'mae': mean_absolute_error(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'r2': r2_score(y_train, y_pred_train)
        },
        'test': {
            'mae': mean_absolute_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'r2': r2_score(y_test, y_pred_test)
        }
    }
    
    # Print metrics
    print("\n📊 Model Performance:")
    print("=" * 50)
    print(f"{'Metric':<15} {'Training':<15} {'Test':<15}")
    print("-" * 50)
    print(f"{'MAE (°C)':<15} {metrics['train']['mae']:<15.3f} {metrics['test']['mae']:<15.3f}")
    print(f"{'RMSE (°C)':<15} {metrics['train']['rmse']:<15.3f} {metrics['test']['rmse']:<15.3f}")
    print(f"{'R² Score':<15} {metrics['train']['r2']:<15.3f} {metrics['test']['r2']:<15.3f}")
    print("=" * 50)
    
    return model, metrics

def save_model_and_scaler(model, scaler, metrics):
    """
    Save the trained model and scaler to disk.
    """
    print("💾 Saving model and scaler...")
    
    # Save model with compression
    with open('model_rf.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save metrics for reference
    with open('model_metrics.json', 'w') as f:
        import json
        json.dump(metrics, f, indent=2)
    
    # Print model info
    print(f"✅ Model saved: {os.path.getsize('model_rf.pkl') / 1024 / 1024:.2f} MB")
    print(f"✅ Scaler saved: {os.path.getsize('scaler.pkl') / 1024:.2f} KB")
    
    # Print feature importance summary
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Get top 10 features
        indices = np.argsort(importances)[::-1][:10]
        
        print("\n🏆 Top 10 Feature Importances:")
        print("-" * 40)
        for i, idx in enumerate(indices):
            print(f"{i+1:2d}. Feature {idx:3d}: {importances[idx]:.4f}")
        print("-" * 40)

def main():
    """Main training pipeline."""
    print("=" * 60)
    print("🌡️ WEATHER TEMPERATURE FORECASTING - MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Generate/Load data
    df = generate_synthetic_data(n_samples=5000)  # Reduced for faster training
    
    # Step 2: Create lag features
    df_lag = create_lag_features(df, lag_hours=24)
    
    # Step 3: Prepare features and target
    X = df_lag.drop(columns=['Temperature (C)'])
    y = df_lag['Temperature (C)']
    
    print(f"\n📐 Dataset shape: {X.shape}")
    print(f"🎯 Target shape: {y.shape}")
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # No shuffle for time series
    )
    
    print(f"\n📊 Data Split:")
    print(f"  Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Step 5: Scale features
    print("\n⚖️ Scaling features...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features scaled")
    
    # Step 6: Train model
    model, metrics = train_random_forest(
        X_train_scaled, y_train, 
        X_test_scaled, y_test
    )
    
    # Step 7: Save model
    save_model_and_scaler(model, scaler, metrics)
    
    # Step 8: Generate sample predictions
    print("\n🔍 Sample Predictions:")
    print("-" * 40)
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    for i, idx in enumerate(sample_indices):
        actual = y_test.iloc[idx]
        predicted = model.predict([X_test_scaled[idx]])[0]
        error = predicted - actual
        print(f"Sample {i+1}: Actual={actual:.2f}°C, "
              f"Predicted={predicted:.2f}°C, "
              f"Error={error:+.2f}°C")
    
    print("\n" + "=" * 60)
    print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n📁 Files generated:")
    print("  - model_rf.pkl (trained model)")
    print("  - scaler.pkl (fitted scaler)")
    print("  - model_metrics.json (performance metrics)")
    
    print("\n🚀 Next steps:")
    print("  1. Run the app: streamlit run app.py")
    print("  2. Deploy to Streamlit Cloud")
    print("  3. Test predictions in the web interface")

if __name__ == "__main__":
>>>>>>> daaa10d4419157170c86b89ec1525bb95e96ba73
    main()