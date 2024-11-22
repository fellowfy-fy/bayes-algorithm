import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def generate_frequency_data(start_date='2018-09-04', end_date='2019-03-23', 
                          f1_mean=7.12, f2_mean=10.86,
                          f1_std=0.05, f2_std=0.07,
                          measurements_per_day=24,
                          acceleration_amplitude=1.0, acceleration_noise=0.1):
    """
    Generate test frequency and acceleration data similar to the paper's measurements
    
    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    f1_mean (float): Mean value for first frequency
    f2_mean (float): Mean value for second frequency
    f1_std (float): Standard deviation for first frequency
    f2_std (float): Standard deviation for second frequency
    measurements_per_day (int): Number of measurements per day
    acceleration_amplitude (float): Base amplitude for acceleration data
    acceleration_noise (float): Standard deviation for acceleration noise
    
    Returns:
    pandas.DataFrame: Generated frequency and acceleration data
    """
    
    # Convert dates to datetime
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Calculate number of days
    days = (end - start).days + 1
    
    # Generate timestamps
    timestamps = []
    for day in range(days):
        current_date = start + timedelta(days=day)
        for hour in range(measurements_per_day):
            timestamps.append(current_date + timedelta(hours=hour))
    
    # Generate frequency data
    total_measurements = len(timestamps)
    seasonal_effect = np.sin(np.linspace(0, 2*np.pi, total_measurements))
    
    f1_base = np.random.normal(f1_mean, f1_std, total_measurements)
    f2_base = np.random.normal(f2_mean, f2_std, total_measurements)
    
    f1 = f1_base + 0.02 * seasonal_effect + 0.01 * np.sin(np.linspace(0, 2*np.pi*days, total_measurements))
    f2 = f2_base + 0.03 * seasonal_effect + 0.015 * np.sin(np.linspace(0, 2*np.pi*days, total_measurements))
    
    # Generate acceleration data
    time_intervals = np.linspace(0, 2 * np.pi * total_measurements, total_measurements)
    acceleration = acceleration_amplitude * np.sin(time_intervals) + np.random.normal(0, acceleration_noise, total_measurements)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': timestamps,
        'f1': f1,
        'f2': f2,
        'acceleration': acceleration
    })
    
    return df

# Generate data
df = generate_frequency_data()

# Save to CSV
csv_filename = 'test_frequency_data_with_acceleration.csv'
df.to_csv(csv_filename, index=False, date_format='%Y-%m-%d %H:%M:%S')

print(f"Generated {len(df)} measurements and saved to {csv_filename}")

# Display first few rows
print("\nFirst few rows of generated data:")
print(df.head())

# Display basic statistics
print("\nBasic statistics of generated data:")
print(df.describe())

# Plot the generated data
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['f1'], label='f1', alpha=0.7)
plt.plot(df['Date'], df['f2'], label='f2', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Frequency (Hz)')
plt.title('Generated Frequency Data')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['acceleration'], label='Acceleration', alpha=0.7, color='orange')
plt.xlabel('Date')
plt.ylabel('Acceleration (m/s²)')
plt.title('Generated Acceleration Data')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
