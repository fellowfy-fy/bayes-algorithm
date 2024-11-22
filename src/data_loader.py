import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import numpy as np

class FrequencyDataLoader:
    def __init__(self):
        """
        Initialize the data loader for multiple files
        """
        self.files_data = {}  # Dictionary to store data from multiple files
        self.combined_data = None  # Combined data from all files
        
    def select_files(self):
        """
        Open file dialog to select one or multiple CSV files
        
        Returns:
        list: List of selected file paths
        """
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        file_paths = filedialog.askopenfilenames(
            title='Select CSV Files',
            filetypes=[('CSV Files', '*.csv'), ('All Files', '*.*')]
        )
        
        return file_paths
    
    def load_data(self, file_paths=None):
        """
        Load data from multiple CSV files
        
        Parameters:
        file_paths (list): Optional list of file paths. If None, opens file dialog
        """
        if file_paths is None:
            file_paths = self.select_files()
            
        if not file_paths:
            print("No files selected")
            return False
        
        for file_path in file_paths:
            try:
                # Read CSV file
                data = pd.read_csv(file_path)
                
                # Convert date column to datetime
                data['Date'] = pd.to_datetime(data['Date'])
                
                # Sort data by date
                data = data.sort_values('Date')
                
                # Store data with filename as key
                filename = Path(file_path).stem
                self.files_data[filename] = data
                
                print(f"Loaded {len(data)} measurements from {filename}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        # Combine all data
        if self.files_data:
            self.combine_data()
            return True
        return False
    
    def combine_data(self):
        """
        Combine data from all loaded files
        """
        if not self.files_data:
            return
            
        # Concatenate all dataframes
        combined = pd.concat(self.files_data.values(), ignore_index=True)
        
        # Sort by date and remove duplicates
        self.combined_data = combined.sort_values('Date').drop_duplicates(subset='Date')
    
    def get_frequency_stats(self, filename=None):
        """
        Calculate basic statistics for both frequencies
        
        Parameters:
        filename (str): Optional filename to get stats for specific file
        
        Returns:
        dict: Dictionary containing mean, std, min, max for both frequencies
        """
        data = self.files_data.get(filename) if filename else self.combined_data
        
        if data is None:
            return None
            
        stats = {
            'f1_mean': data['f1'].mean(),
            'f1_std': data['f1'].std(),
            'f1_min': data['f1'].min(),
            'f1_max': data['f1'].max(),
            'f2_mean': data['f2'].mean(),
            'f2_std': data['f2'].std(),
            'f2_min': data['f2'].min(),
            'f2_max': data['f2'].max()
        }
        return stats
    
    def plot_frequencies(self, filename=None):
        """
        Create a time series plot of both frequencies
        
        Parameters:
        filename (str): Optional filename to plot specific file
        """
        if filename:
            data = self.files_data.get(filename)
            title_suffix = f" - {filename}"
        else:
            data = self.combined_data
            title_suffix = " - All Files Combined"
            
        if data is None:
            return
            
        plt.figure(figsize=(12,6))
        plt.plot(data['Date'], data['f1'], label='f1', alpha=0.7)
        plt.plot(data['Date'], data['f2'], label='f2', alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('Frequency')
        plt.title(f'Frequency Measurements Over Time{title_suffix}')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_all_files(self):
        """
        Plot data from all files in separate subplots
        """
        if not self.files_data:
            return
            
        n_files = len(self.files_data)
        fig, axes = plt.subplots(n_files, 1, figsize=(12, 6*n_files))
        
        if n_files == 1:
            axes = [axes]
            
        for ax, (filename, data) in zip(axes, self.files_data.items()):
            ax.plot(data['Date'], data['f1'], label='f1', alpha=0.7)
            ax.plot(data['Date'], data['f2'], label='f2', alpha=0.7)
            ax.set_xlabel('Date')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Frequency Measurements - {filename}')
            ax.legend()
            ax.grid(True)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def get_monthly_data(self, filename=None):
        """
        Group data by month and return monthly averages
        
        Parameters:
        filename (str): Optional filename to get monthly data for specific file
        
        Returns:
        pandas.DataFrame: Monthly averaged frequency data
        """
        data = self.files_data.get(filename) if filename else self.combined_data
        
        if data is None:
            return None
            
        monthly_data = data.set_index('Date').resample('M').mean()
        return monthly_data

# Example usage
if __name__ == "__main__":
    loader = FrequencyDataLoader()
    
    if loader.load_data():
        # Print statistics for all files combined
        print("\nCombined Statistics:")
        stats = loader.get_frequency_stats()
        for key, value in stats.items():
            print(f"{key}: {value:.4f}")
        
        # Plot individual files
        loader.plot_all_files()
        
        # Plot combined data
        loader.plot_frequencies()
        
        # Get monthly averages for combined data
        monthly_data = loader.get_monthly_data()
        print("\nMonthly Averages (Combined Data):")
        print(monthly_data)
        
        # Print statistics for each file
        for filename in loader.files_data.keys():
            print(f"\nStatistics for {filename}:")
            stats = loader.get_frequency_stats(filename)
            for key, value in stats.items():
                print(f"{key}: {value:.4f}")