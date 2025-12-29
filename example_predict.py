"""
Example script for testing predict.py

This script demonstrates how to use the predictive maintenance API
with real data examples from the test dataset.
"""

import json
import numpy as np
import pandas as pd
import requests
from typing import List, Dict
from train import load_data, create_test_windows

# Configuration
WINDOW_SIZE = 20
INPUT_SIZE = 17
API_URL = "http://localhost:8080/predict"

# Sensors removed during training (constant sensors)
CONSTANT_SENSORS = ['sensor_1', 'sensor_6', 'sensor_5', 'sensor_10',
                    'sensor_16', 'sensor_18', 'sensor_19']


def prepare_features_from_dataframe(df: pd.DataFrame, engine_id: int) -> List[float]:
    """
    Prepare features from a dataframe for a specific engine.
    
    Args:
        df: DataFrame with engine data
        engine_id: Engine ID to extract features for
        
    Returns:
        List of 340 features (20 timesteps × 17 features)
    """
    # Filter data for specific engine
    engine_df = df[df['engine_id'] == engine_id].sort_values('cycle')
    
    if len(engine_df) < WINDOW_SIZE:
        raise ValueError(f"Engine {engine_id} has only {len(engine_df)} cycles, need at least {WINDOW_SIZE}")
    
    # Get the last WINDOW_SIZE cycles
    engine_df = engine_df.tail(WINDOW_SIZE)
    
    # Define feature columns (3 operational settings + sensors, excluding constant ones)
    feature_cols = ['op_set_1', 'op_set_2', 'op_set_3']
    sensor_cols = [f"sensor_{i}" for i in range(1, 22) if f"sensor_{i}" not in CONSTANT_SENSORS]
    feature_cols.extend(sensor_cols)
    
    # Extract features
    features = engine_df[feature_cols].values  # Shape: (20, 17)
    
    # Flatten to list: 20 × 17 = 340 values
    return features.flatten().tolist()


def example_1_load_from_test_file():
    """
    Example 1: Load data from test file and make predictions
    """
    print("\n" + "="*60)
    print("Example 1: Loading data from test file")
    print("="*60)
    
    # Load test data
    test_path = "data/test_FD001.txt"
    test_df = load_data(test_path)
    
    print(f"Loaded {len(test_df)} rows from {test_path}")
    print(f"Unique engines: {test_df['engine_id'].nunique()}")
    
    # Get first few engines
    engine_ids = test_df['engine_id'].unique()[:3]
    
    for engine_id in engine_ids:
        try:
            # Prepare features
            features = prepare_features_from_dataframe(test_df, engine_id)
            
            # Make prediction
            response = requests.post(API_URL, json={"features": features})
            
            if response.status_code == 200:
                result = response.json()
                print(f"\nEngine {engine_id}:")
                print(f"  Predicted Class: {result['predicted_class']} "
                      f"({'Failure Imminent' if result['predicted_class'] == 1 else 'Normal'})")
                print(f"  Confidence: {result['confidence']:.4f}")
            else:
                print(f"\nEngine {engine_id}: Error - {response.status_code}")
                print(f"  {response.text}")
        except Exception as e:
            print(f"\nEngine {engine_id}: Error - {str(e)}")


def example_2_random_data():
    """
    Example 2: Test with randomly generated data
    """
    print("\n" + "="*60)
    print("Example 2: Testing with random data")
    print("="*60)
    
    # Generate random features
    # Operational settings (0-2): typically in specific ranges
    # Sensors (3-16): will be scaled by the scaler
    random_features = []
    
    for _ in range(WINDOW_SIZE):
        # 3 operational settings (typically 0-1 or normalized)
        op_settings = np.random.uniform(0, 1, 3).tolist()
        # 14 sensor values (will be scaled)
        sensors = np.random.randn(14).tolist()
        random_features.extend(op_settings + sensors)
    
    print(f"Generated {len(random_features)} random features")
    
    try:
        response = requests.post(API_URL, json={"features": random_features})
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nRandom Data Prediction:")
            print(f"  Predicted Class: {result['predicted_class']} "
                  f"({'Failure Imminent' if result['predicted_class'] == 1 else 'Normal'})")
            print(f"  Confidence: {result['confidence']:.4f}")
        else:
            print(f"Error - {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")


def example_3_simulate_engine_degradation():
    """
    Example 3: Simulate engine degradation over time
    """
    print("\n" + "="*60)
    print("Example 3: Simulating engine degradation")
    print("="*60)
    
    # Simulate an engine that starts normal and degrades
    base_features = []
    
    # Start with normal operating conditions
    for timestep in range(WINDOW_SIZE):
        # Operational settings (stable)
        op_settings = [0.5, 0.3, 0.7]
        
        # Sensors start normal, gradually increase (simulating degradation)
        degradation_factor = 1.0 + (timestep / WINDOW_SIZE) * 0.5
        sensors = np.random.randn(14) * degradation_factor
        
        base_features.extend(op_settings + sensors.tolist())
    
    print(f"Simulated degradation pattern (increasing sensor values)")
    
    try:
        response = requests.post(API_URL, json={"features": base_features})
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nDegraded Engine Prediction:")
            print(f"  Predicted Class: {result['predicted_class']} "
                  f"({'Failure Imminent' if result['predicted_class'] == 1 else 'Normal'})")
            print(f"  Confidence: {result['confidence']:.4f}")
        else:
            print(f"Error - {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")


def example_4_batch_predictions():
    """
    Example 4: Make batch predictions for multiple engines
    """
    print("\n" + "="*60)
    print("Example 4: Batch predictions for multiple engines")
    print("="*60)
    
    # Load test data
    test_path = "data/test_FD001.txt"
    test_df = load_data(test_path)
    
    # Get first 5 engines
    engine_ids = test_df['engine_id'].unique()[:5]
    
    results = []
    
    for engine_id in engine_ids:
        try:
            features = prepare_features_from_dataframe(test_df, engine_id)
            response = requests.post(API_URL, json={"features": features})
            
            if response.status_code == 200:
                result = response.json()
                results.append({
                    'engine_id': engine_id,
                    'predicted_class': result['predicted_class'],
                    'confidence': result['confidence']
                })
            else:
                print(f"Engine {engine_id}: Failed with status {response.status_code}")
        except Exception as e:
            print(f"Engine {engine_id}: Error - {str(e)}")
    
    # Print summary
    print(f"\nBatch Prediction Results ({len(results)} engines):")
    print("-" * 60)
    for r in results:
        status = "Failure Imminent" if r['predicted_class'] == 1 else "Normal"
        print(f"Engine {r['engine_id']:3d}: {status:20s} (confidence: {r['confidence']:.4f})")
    
    # Statistics
    if results:
        failure_count = sum(1 for r in results if r['predicted_class'] == 1)
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"\nSummary:")
        print(f"  Engines with predicted failure: {failure_count}/{len(results)}")
        print(f"  Average confidence: {avg_confidence:.4f}")


def example_5_direct_model_usage():
    """
    Example 5: Use the model directly without API (for testing)
    """
    print("\n" + "="*60)
    print("Example 5: Direct model usage (without API)")
    print("="*60)
    
    try:
        import torch
        import pickle
        from predict import model, scaler, WINDOW_SIZE, INPUT_SIZE
        
        # Generate sample data
        sample_features = np.random.randn(WINDOW_SIZE, INPUT_SIZE).astype(np.float32)
        
        # Scale sensor columns (indices 3-16)
        sensor_indices = list(range(3, 17))
        op_set_indices = list(range(3))
        
        x_sensors = scaler.transform(sample_features[:, sensor_indices])
        x_scaled = np.zeros_like(sample_features)
        x_scaled[:, op_set_indices] = sample_features[:, op_set_indices]
        x_scaled[:, sensor_indices] = x_sensors
        
        # Convert to tensor
        x = torch.tensor(x_scaled).unsqueeze(0)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        print(f"\nDirect Model Prediction:")
        print(f"  Predicted Class: {pred_class} "
              f"({'Failure Imminent' if pred_class == 1 else 'Normal'})")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Class probabilities: Normal={probs[0,0]:.4f}, Failure={probs[0,1]:.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Note: This example requires the model and scaler to be loaded")


def example_6_save_request_example():
    """
    Example 6: Save a sample request to JSON file
    """
    print("\n" + "="*60)
    print("Example 6: Saving sample request to JSON")
    print("="*60)
    
    # Generate sample features
    sample_features = []
    for _ in range(WINDOW_SIZE):
        op_settings = [0.5, 0.3, 0.7]
        sensors = np.random.randn(14).tolist()
        sample_features.extend(op_settings + sensors)
    
    # Create request payload
    request_data = {"features": sample_features}
    
    # Save to file
    output_file = "request_example.json"
    with open(output_file, "w") as f:
        json.dump(request_data, f, indent=2)
    
    print(f"Saved sample request to {output_file}")
    print(f"Total features: {len(sample_features)}")
    print(f"\nYou can use this file to test the API:")
    print(f"  curl -X POST http://localhost:8080/predict \\")
    print(f"       -H 'Content-Type: application/json' \\")
    print(f"       -d @{output_file}")


def main():
    """
    Run all examples
    """
    print("\n" + "="*60)
    print("Predictive Maintenance API - Example Tests")
    print("="*60)
    print("\nNote: Make sure the API server is running:")
    print("  python predict.py")
    print("\nOr start it with:")
    print("  uvicorn predict:app --host 0.0.0.0 --port 8080")
    
    # Check if API is available
    try:
        response = requests.get("http://localhost:8080/docs", timeout=2)
        print("\n✅ API server is running")
    except:
        print("\n⚠️  Warning: API server may not be running at http://localhost:8080")
        print("   Some examples may fail. Start the server first.")
    
    # Run examples
    try:
        example_1_load_from_test_file()
    except Exception as e:
        print(f"\nExample 1 failed: {str(e)}")
    
    try:
        example_2_random_data()
    except Exception as e:
        print(f"\nExample 2 failed: {str(e)}")
    
    try:
        example_3_simulate_engine_degradation()
    except Exception as e:
        print(f"\nExample 3 failed: {str(e)}")
    
    try:
        example_4_batch_predictions()
    except Exception as e:
        print(f"\nExample 4 failed: {str(e)}")
    
    try:
        example_5_direct_model_usage()
    except Exception as e:
        print(f"\nExample 5 failed: {str(e)}")
    
    try:
        example_6_save_request_example()
    except Exception as e:
        print(f"\nExample 6 failed: {str(e)}")
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()

