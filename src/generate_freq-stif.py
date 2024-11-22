import pandas as pd
import numpy as np

def generate_test_data(output_csv, stiffness_min=20, stiffness_max=60, num_points=30, k=0.1, noise_level=0.05):
    """
    Generate test data for surrogate model calibration.

    Parameters:
    - output_csv (str): Path to save generated test data.
    - stiffness_min (float): Minimum stiffness value.
    - stiffness_max (float): Maximum stiffness value.
    - num_points (int): Number of stiffness values to generate.
    - k (float): Coefficient to simulate frequency as a function of stiffness.
    - noise_level (float): Level of random noise to add to the frequencies.

    Returns:
    - None
    """
    # Generate stiffness values
    stiffness_values = np.linspace(stiffness_min, stiffness_max, num_points)
    
    # Generate frequencies with noise
    frequencies = [k * np.sqrt(stiffness) + np.random.uniform(-noise_level, noise_level) for stiffness in stiffness_values]
    
    # Create DataFrame
    test_data = pd.DataFrame({
        'stiffness': stiffness_values,
        'frequency': frequencies
    })
    
    # Save to CSV
    test_data.to_csv(output_csv, index=False)
    print(f"Generated test data saved to {output_csv}")

# Example usage
output_csv_path = 'generated_test_data.csv'
generate_test_data(output_csv_path)
