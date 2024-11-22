import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

class FrequencyDataGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Frequency Data Processor")
        self.root.geometry("600x600")
        
        # Data storage
        self.frequency_data = None
        
        # Variables for month and year selection
        self.selected_year = tk.StringVar()
        self.selected_month = tk.StringVar()
        
        self.create_widgets()
    
    def create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Buttons Section
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Button(button_frame, text="Load Frequency Data", command=self.load_frequency_data).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Show Filtered Graph", command=self.plot_frequency_data).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Show All Data", command=self.plot_all_data).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Run Stiffness Calibration", command=self.run_stiffness_calibration).grid(row=0, column=3, padx=5)
        
        # Month and Year Selection Section
        filter_frame = ttk.LabelFrame(main_frame, text="Filter by Month and Year", padding="10")
        filter_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(filter_frame, text="Year:").grid(row=0, column=0, sticky=tk.W)
        self.year_combobox = ttk.Combobox(filter_frame, textvariable=self.selected_year, state='readonly')
        self.year_combobox.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(filter_frame, text="Month:").grid(row=1, column=0, sticky=tk.W)
        self.month_combobox = ttk.Combobox(filter_frame, textvariable=self.selected_month, state='readonly')
        self.month_combobox['values'] = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        self.month_combobox.grid(row=1, column=1, padx=5, pady=2)
        
        # Stiffness Calibration Section
        calibration_frame = ttk.LabelFrame(main_frame, text="Stiffness Calibration", padding="10")
        calibration_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(calibration_frame, text="Base Stiffness:").grid(row=0, column=0, sticky=tk.W)
        self.base_stiffness = tk.DoubleVar()
        ttk.Entry(calibration_frame, textvariable=self.base_stiffness).grid(row=0, column=1, padx=5, pady=2)
        
        # Surrogate Model Calibration Section
        surrogate_frame = ttk.LabelFrame(main_frame, text="Surrogate Model Calibration", padding="10")
        surrogate_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Button(surrogate_frame, text="Load Surrogate Data", command=self.load_surrogate_data).grid(row=0, column=0, padx=5)
        ttk.Button(surrogate_frame, text="Run Surrogate Calibration", command=self.run_surrogate_calibration).grid(row=0, column=1, padx=5)
        
        # Buttons for updating the surrogate model
        update_frame = ttk.LabelFrame(main_frame, text="Update Surrogate Model", padding="10")
        update_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=10)

        ttk.Button(update_frame, text="Receive New Data", command=self.receive_new_data).grid(row=0, column=0, padx=5)
        
        # Status text (move to the bottom)
        self.status_text = tk.Text(main_frame, height=10, width=70)
        self.status_text.grid(row=5, column=0, pady=10, sticky=(tk.W, tk.E))


    
    def load_frequency_data(self):
        """Load frequency data from file"""
        file_path = filedialog.askopenfilename(
            title='Select Frequency Data File',
            filetypes=[('CSV Files', '*.csv'), ('All Files', '*.*')]
        )
        
        if file_path:
            try:
                # Load the data
                self.frequency_data = pd.read_csv(file_path, parse_dates=['Date'])
                
                # Validate required columns
                required_columns = ['Date', 'f1', 'f2']
                missing_columns = [col for col in required_columns if col not in self.frequency_data.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
                
                # Populate year options
                years = self.frequency_data['Date'].dt.year.unique()
                self.year_combobox['values'] = sorted(years)
                
                self.status_text.insert(tk.END, f"Loaded frequency data from {file_path}\n")
                self.status_text.insert(tk.END, f"Number of records: {len(self.frequency_data)}\n")
            
            except ValueError as ve:
                self.status_text.insert(tk.END, f"Error: {ve}\n")
            except Exception as e:
                self.status_text.insert(tk.END, f"Error loading file: {str(e)}\n")
    
    def plot_all_data(self):
        """Plot the frequency data for all time"""
        if self.frequency_data is None:
            self.status_text.insert(tk.END, "No frequency data loaded\n")
            return
        
        # Plot all data
        plt.figure(figsize=(12, 6))
        plt.plot(self.frequency_data['Date'], self.frequency_data['f1'], label='f1', alpha=0.7)
        plt.plot(self.frequency_data['Date'], self.frequency_data['f2'], label='f2', alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('Frequency (Hz)')
        plt.title("Frequencies Over All Time")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_frequency_data(self):
        """Plot the frequency data filtered by month and year"""
        if self.frequency_data is None:
            self.status_text.insert(tk.END, "No frequency data loaded\n")
            return
        
        # Filter data by selected year and month
        year = self.selected_year.get()
        month = self.selected_month.get()
        
        if not year or not month:
            self.status_text.insert(tk.END, "Please select both year and month\n")
            return
        
        # Convert month name to number
        month_number = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
            'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10,
            'November': 11, 'December': 12
        }[month]
        
        # Filter data
        filtered_data = self.frequency_data[
            (self.frequency_data['Date'].dt.year == int(year)) &
            (self.frequency_data['Date'].dt.month == month_number)
        ]
        
        if filtered_data.empty:
            self.status_text.insert(tk.END, f"No data found for {month} {year}\n")
            return
        
        # Plot filtered data
        plt.figure(figsize=(12, 6))
        plt.plot(filtered_data['Date'], filtered_data['f1'], label='f1', alpha=0.7)
        plt.plot(filtered_data['Date'], filtered_data['f2'], label='f2', alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('Frequency (Hz)')
        plt.title(f"Frequencies in {month} {year}")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def run_stiffness_calibration(self):
        """Run calibration by modifying stiffness and calculating frequencies"""
        if self.frequency_data is None:
            self.status_text.insert(tk.END, "No frequency data loaded\n")
            return

        base_stiffness = self.base_stiffness.get()
        if not base_stiffness:
            self.status_text.insert(tk.END, "Please enter a base stiffness value\n")
            return

        # Calculate stiffness variations
        stiffness_variations = {
            "60% Stiffness": base_stiffness * 0.6,
            "Base Stiffness": base_stiffness,
            "140% Stiffness": base_stiffness * 1.4
        }

        # Simulate new frequencies based on stiffness
        frequencies = {}
        for key, stiffness in stiffness_variations.items():
            f1_new = self.frequency_data['f1'] * (stiffness / base_stiffness) ** 0.5
            f2_new = self.frequency_data['f2'] * (stiffness / base_stiffness) ** 0.5
            frequencies[key] = (f1_new, f2_new)

        # Plot results
        plt.figure(figsize=(12, 6))
        for key, (f1_new, f2_new) in frequencies.items():
            plt.plot(self.frequency_data['Date'], f1_new, label=f'f1 ({key})', alpha=0.7)
            plt.plot(self.frequency_data['Date'], f2_new, label=f'f2 ({key})', alpha=0.7)

        plt.xlabel('Date')
        plt.ylabel('Frequency (Hz)')
        plt.title('Effect of Stiffness Variations on Frequencies')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        self.status_text.insert(tk.END, "Stiffness calibration completed and plot displayed.\n")
    
    def load_surrogate_data(self):
        """Load surrogate model calibration data from a file"""
        file_path = filedialog.askopenfilename(
            title='Select Surrogate Data File',
            filetypes=[('CSV Files', '*.csv'), ('All Files', '*.*')]
        )
        
        if file_path:
            try:
                # Load the data
                self.surrogate_data = pd.read_csv(file_path)
                
                # Validate required columns
                required_columns = ['stiffness', 'frequency']
                missing_columns = [col for col in required_columns if col not in self.surrogate_data.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
                
                # Initialize the model
                from sklearn.linear_model import LinearRegression
                self.model = LinearRegression()
                
                # Fit the initial surrogate model
                stiffness = self.surrogate_data['stiffness'].values.reshape(-1, 1)
                frequency = self.surrogate_data['frequency'].values
                self.model.fit(stiffness, frequency)
                
                self.status_text.insert(tk.END, f"Loaded surrogate model data from {file_path}\n")
                self.status_text.insert(tk.END, f"Number of records: {len(self.surrogate_data)}\n")
            
            except ValueError as ve:
                self.status_text.insert(tk.END, f"Error: {ve}\n")
            except Exception as e:
                self.status_text.insert(tk.END, f"Error loading file: {str(e)}\n")

    def run_surrogate_calibration(self):
        """Run surrogate model calibration and plot results using uploaded data"""
        if not hasattr(self, 'surrogate_data') or self.surrogate_data is None:
            self.status_text.insert(tk.END, "No surrogate model data loaded\n")
            return

        try:
            # Extract stiffness and frequency from the loaded data
            stiffness = self.surrogate_data['stiffness'].values.reshape(-1, 1)  # Reshape for sklearn
            frequency = self.surrogate_data['frequency'].values
            
            # Fit surrogate model (e.g., linear regression)
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(stiffness, frequency)
            predicted_frequency = model.predict(stiffness)

            # Plot the data and model
            plt.figure(figsize=(12, 6))
            plt.scatter(self.surrogate_data['stiffness'], self.surrogate_data['frequency'], 
                        label='MKЭ Data', color='blue', alpha=0.7)
            plt.plot(self.surrogate_data['stiffness'], predicted_frequency, 
                    label='Surrogate Model', color='red', linewidth=2)
            plt.xlabel('Stiffness')
            plt.ylabel('Frequency')
            plt.title('Surrogate Model Calibration')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            self.status_text.insert(tk.END, "Surrogate model calibration completed and plot displayed.\n")
        
        except Exception as e:
            self.status_text.insert(tk.END, f"Error during surrogate calibration: {str(e)}\n")

    def update_surrogate_model(self, new_stiffness, new_frequency):
        """Update the surrogate model with new data points"""
        try:
            # Append new data to the existing surrogate data
            new_data = pd.DataFrame({'stiffness': [new_stiffness], 'frequency': [new_frequency]})
            self.surrogate_data = pd.concat([self.surrogate_data, new_data], ignore_index=True)

            # Refit the surrogate model with the updated data
            stiffness = self.surrogate_data['stiffness'].values.reshape(-1, 1)
            frequency = self.surrogate_data['frequency'].values

            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
            self.model.fit(stiffness, frequency)

            self.status_text.insert(tk.END, f"Model updated with new data: stiffness={new_stiffness}, frequency={new_frequency}\n")
        
        except Exception as e:
            self.status_text.insert(tk.END, f"Error during model update: {str(e)}\n")

    def receive_new_data(self):
        """Simulate receiving multiple new data points and updating the Bayesian model"""
        if not hasattr(self, 'surrogate_data') or self.surrogate_data is None:
            self.status_text.insert(tk.END, "No surrogate model data loaded to update\n")
            return

        if not hasattr(self, 'bayesian_trace') or self.bayesian_trace is None:
            self.status_text.insert(tk.END, "Bayesian model not initialized. Run surrogate calibration first.\n")
            return

        try:
            # Simulate new data
            num_points = 10
            new_stiffness_values = np.random.uniform(20, 60, num_points)
            alpha_mean = self.bayesian_trace.posterior['alpha'].mean().item()
            beta_mean = self.bayesian_trace.posterior['beta'].mean().item()
            new_frequency_values = (
                alpha_mean + beta_mean * new_stiffness_values +
                np.random.uniform(-0.1, 0.1, num_points)  # Adding noise
            )

            # Add new data to surrogate dataset
            new_data = pd.DataFrame({'stiffness': new_stiffness_values, 'frequency': new_frequency_values})
            self.surrogate_data = pd.concat([self.surrogate_data, new_data], ignore_index=True)

            # Refit the Bayesian model with updated data
            stiffness = self.surrogate_data['stiffness'].values
            frequency = self.surrogate_data['frequency'].values
            self.bayesian_model, self.bayesian_trace = fit_bayesian_model(stiffness, frequency)

            # Log the added data
            self.status_text.insert(
                tk.END, f"Added {num_points} new data points. Bayesian model updated.\n"
            )

            # Plot posterior distributions
            import arviz as az
            az.plot_posterior(self.bayesian_trace)
            plt.show()
        
        except Exception as e:
            self.status_text.insert(tk.END, f"Error receiving new data: {str(e)}\n")



    def run_surrogate_calibration(self):
        """Run Bayesian surrogate model calibration and plot results"""
        if not hasattr(self, 'surrogate_data') or self.surrogate_data is None:
            self.status_text.insert(tk.END, "No surrogate model data loaded\n")
            return

        try:
            # Prepare stiffness and frequency for the model
            stiffness = self.surrogate_data['stiffness'].values
            frequency = self.surrogate_data['frequency'].values

            # Fit Bayesian model
            self.bayesian_model, self.bayesian_trace = fit_bayesian_model(stiffness, frequency)

            # Plot posterior distributions
            import arviz as az
            az.plot_posterior(self.bayesian_trace)
            plt.show()

            self.status_text.insert(tk.END, "Bayesian surrogate model calibration completed.\n")
        
        except Exception as e:
            self.status_text.insert(tk.END, f"Error during Bayesian model calibration: {str(e)}\n")



    def plot_surrogate_model(self):
        """Plot the current surrogate model"""
        try:
            stiffness = self.surrogate_data['stiffness'].values.reshape(-1, 1)
            frequency = self.surrogate_data['frequency']
            predicted_frequency = self.model.predict(stiffness)

            plt.figure(figsize=(12, 6))
            plt.scatter(self.surrogate_data['stiffness'], self.surrogate_data['frequency'], 
                        label='MKE Data', color='blue', alpha=0.7)
            plt.plot(self.surrogate_data['stiffness'], predicted_frequency, 
                    label='Surrogate Model', color='red', linewidth=2)
            plt.xlabel('Stiffness')
            plt.ylabel('Frequency')
            plt.title('Surrogate Model Calibration (Updated)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            self.status_text.insert(tk.END, f"Error during plotting: {str(e)}\n")


    def run(self):
        self.root.mainloop()


def fit_bayesian_model(stiffness, frequency):
    """
    Fit Bayesian linear regression model using PyMC3.
    
    Parameters:
    - stiffness: Array of stiffness values (input feature).
    - frequency: Array of frequency values (target variable).
    
    Returns:
    - model: PyMC3 model object.
    - trace: Posterior samples from the model.
    """
    with pm.Model() as model:
        # Priors for parameters
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)
        
        # Linear model
        mu = alpha + beta * stiffness
        
        # Likelihood
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=frequency)
        
        # Posterior sampling
        trace = pm.sample(1000, return_inferencedata=True)
    
    return model, trace

if __name__ == "__main__":
    app = FrequencyDataGUI()
    app.run()
