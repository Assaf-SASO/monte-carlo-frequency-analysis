import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import pandas as pd
import matplotlib as mpl
from datetime import datetime
from scipy.stats import norm, skewnorm, t, cauchy

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12

# Seed 42 was picked for reproducibility
np.random.seed(42)

# Number of iterations (Can be changed)
iterations = 400000

# Base frequency (Based on the model)
fm = 50000000  # Hz

# Parameters from the uncertainty budget (Based on the model)
standard_uncertainty = np.random.normal(0, 0.018371382, iterations)
random_uncertainty = np.random.normal(0, 2.095882062343330E-01, iterations)
display_resolution = np.random.uniform(-5.0e-15, 5.0e-15, iterations)
time_base_accuracy = np.random.normal(0, 0.0105303846083607000, iterations)
systematic_uncertainty = np.random.uniform(-1.0e-07, 1.0e-07, iterations)

# Calculate frequency using the model
# f = fm(1 + display + systematic) + standard + random + timebase
frequency = fm * (1 + display_resolution + systematic_uncertainty) + standard_uncertainty + random_uncertainty + time_base_accuracy

# Statistics 
mean = np.mean(frequency)
std_dev = np.std(frequency)
skewness = stats.skew(frequency)
kurtosis = stats.kurtosis(frequency)

# Calculate contribution of each component to the total variance
# We need to calculate the variance of each component's effect on the final result

# Standard uncertainty contribution
standard_contribution = np.var(standard_uncertainty)

# Random uncertainty contribution
random_contribution = np.var(random_uncertainty)

# Display resolution contribution
display_contribution = np.var(fm * display_resolution)

# Time base accuracy contribution
timebase_contribution = np.var(time_base_accuracy)

# Systematic uncertainty contribution
systematic_contribution = np.var(fm * systematic_uncertainty)

# Total variance
total_variance = np.var(frequency)

# Calculate percentage contributions
standard_percentage = (standard_contribution / total_variance) * 100
random_percentage = (random_contribution / total_variance) * 100
display_percentage = (display_contribution / total_variance) * 100
timebase_percentage = (timebase_contribution / total_variance) * 100
systematic_percentage = (systematic_contribution / total_variance) * 100

# Calculate coverage intervals (95% and 99%)
ci_95_lower = np.percentile(frequency, 2.5)
ci_95_upper = np.percentile(frequency, 97.5)
ci_99_lower = np.percentile(frequency, 0.5)
ci_99_upper = np.percentile(frequency, 99.5)

# Calculate uncertainties from coverage intervals
uncertainty_95 = (ci_95_upper - ci_95_lower) / 2
uncertainty_99 = (ci_99_upper - ci_99_lower) / 2

# Calculate GUM uncertainties
unexpanded_uncertainty = std_dev  # unexpanded uncertainty k=1
expanded_uncertainty_k2 = 2 * std_dev  # expanded uncertainty k=2

# Print results
print("\n===== Monte Carlo Simulation Results (400,000 iterations) =====")
print(f"Mean Frequency: {mean:.6f} Hz")
print(f"Standard Deviation: {std_dev:.6f} Hz")
print(f"Skewness: {skewness:.6f}")
print(f"Kurtosis: {kurtosis:.6f}")

print("\n===== Coverage Intervals =====")
print(f"95% Coverage Interval (95% confidence): [{ci_95_lower:.6f}, {ci_95_upper:.6f}] Hz")
print(f"95% Coverage Uncertainty: {uncertainty_95:.6f} Hz")
print(f"95% Result: {mean:.6f} +/- {uncertainty_95:.6f} Hz")

print(f"99% Coverage Interval (99% confidence): [{ci_99_lower:.6f}, {ci_99_upper:.6f}] Hz")
print(f"99% Coverage Uncertainty: {uncertainty_99:.6f} Hz")
print(f"99% Result: {mean:.6f} +/- {uncertainty_99:.6f} Hz")

print("\n===== GUM Approach =====")
print(f"Unexpanded Uncertainty (k=1): {unexpanded_uncertainty:.6f} Hz")
print(f"Expanded Uncertainty (k=2): {expanded_uncertainty_k2:.6f} Hz")
print(f"GUM Result (k=1): {mean:.6f} +/- {unexpanded_uncertainty:.6f} Hz")
print(f"GUM Result (k=2): {mean:.6f} +/- {expanded_uncertainty_k2:.6f} Hz")

print("\n===== Uncertainty Contributions =====")
contributors = [
    ["Standard Uncertainty", standard_percentage],
    ["Random Uncertainty", random_percentage],
    ["Display Resolution", display_percentage],
    ["Time Base Accuracy", timebase_percentage],
    ["Systematic Uncertainty", systematic_percentage]
]

# Sort contributors by percentage (descending)
contributors.sort(key=lambda x: x[1], reverse=True)

# Print table of contributions
print("{:<25} {:>15}".format("Contributor", "Percentage (%)"))
print("-" * 42)
for contributor, percentage in contributors:
    print("{:<25} {:>15.2f}".format(contributor, percentage))


# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create a timestamped folder for this run (each run will have its own folder)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_folder_name = f"monte_carlo_run_{timestamp}"
output_dir = os.path.join(script_dir, run_folder_name)
os.makedirs(output_dir, exist_ok=True)
print(f"\n===== Output Directory =====")
print(f"Created output directory: {output_dir}")

# Export frequency data (each 400k iteration) to a file.
# While this currently uses an Excel (.xlsx) file, it can be changed to more efficient formats like CSV, Parquet, or Feather
# to reduce file size, save disk space, and improve read/write performance.
excel_file_path = os.path.join(output_dir, "monte_carlo_frequency_data.xlsx")

# Create a DataFrame with the frequency data
df = pd.DataFrame({
    'Frequency (Hz)': frequency
})

# Save to Excel
df.to_excel(excel_file_path, index=False)
print(f"\n===== Excel Export =====")
print(f"Frequency data exported to: {excel_file_path}")

# Generate and save professional plots

# 1. Q-Q Plot (Assess Normality)
plt.figure(figsize=(10, 8))
res = stats.probplot(frequency, plot=plt)
plt.title('Q-Q Plot of Monte Carlo Frequency Data', fontweight='bold')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
qq_plot_path = os.path.join(output_dir, "monte_carlo_qq_plot.png")
plt.savefig(qq_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Q-Q Plot saved to: {qq_plot_path}")

# 2. Raw Distribution Plot (without normal curve)
plt.figure(figsize=(12, 8))
sns_color = '#4878CF'  # Seaborn-like blue color
n, bins, patches = plt.hist(frequency, bins=100, density=True, alpha=0.75, color=sns_color)

plt.title('Raw Frequency Distribution from Monte Carlo Simulation', fontweight='bold')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Probability Density')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
raw_dist_plot_path = os.path.join(output_dir, "monte_carlo_raw_distribution.png")
plt.savefig(raw_dist_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Raw Distribution Plot saved to: {raw_dist_plot_path}")

# 3. Distribution Plot with normal curve overlay (Assess Normality)
plt.figure(figsize=(12, 8))
n, bins, patches = plt.hist(frequency, bins=100, density=True, alpha=0.75, color=sns_color)

# Add a best fit line ((only meaningful if the data is approximately normally distributed)
x = np.linspace(min(frequency), max(frequency), 1000)
y = stats.norm.pdf(x, mean, std_dev)
plt.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')

plt.title('Frequency Distribution with Normal Curve', fontweight='bold')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Probability Density')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
dist_plot_path = os.path.join(output_dir, "monte_carlo_distribution_with_normal.png")
plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Distribution Plot with Normal Curve saved to: {dist_plot_path}")

# 4. Distribution Plot with 95% Coverage
plt.figure(figsize=(12, 8))

# Plot histogram
n, bins, patches = plt.hist(frequency, bins=100, density=True, alpha=0.6, color=sns_color)

# Add a best fit line
x = np.linspace(min(frequency), max(frequency), 1000)
y = stats.norm.pdf(x, mean, std_dev)
plt.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')

# Highlight 95% coverage interval
x_fill = np.linspace(ci_95_lower, ci_95_upper, 1000)
y_fill = stats.norm.pdf(x_fill, mean, std_dev)
plt.fill_between(x_fill, y_fill, alpha=0.3, color='green', label='95% Coverage Interval')

# Add vertical lines for the coverage interval boundaries
plt.axvline(x=ci_95_lower, color='green', linestyle='--', alpha=0.8)
plt.axvline(x=ci_95_upper, color='green', linestyle='--', alpha=0.8)

# Add text labels directly on the plot for better readability
# Get the current y-axis limits to position the text appropriately
plt.ylim()  # This forces matplotlib to calculate the y-limits
ymin, ymax = plt.ylim()

# Calculate offsets from base frequency for clearer display
lower_offset = ci_95_lower - fm
upper_offset = ci_95_upper - fm

# Add text labels for the lower and upper bounds inside the plot with offset values
plt.text(ci_95_lower + (ci_95_upper - ci_95_lower)*0.05, ymax*0.85, 
         f'Offset: {lower_offset:.6f} Hz', 
         fontsize=12, color='darkgreen', fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='green', boxstyle='round,pad=0.5'))

plt.text(ci_95_upper - (ci_95_upper - ci_95_lower)*0.25, ymax*0.85, 
         f'Offset: {upper_offset:.6f} Hz', 
         fontsize=12, color='darkgreen', fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='green', boxstyle='round,pad=0.5'))

plt.title('Frequency Distribution with 95% Coverage Interval', fontweight='bold')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Probability Density')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
coverage_plot_path = os.path.join(output_dir, "monte_carlo_distribution_95coverage.png")
plt.savefig(coverage_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Distribution Plot with 95% Coverage saved to: {coverage_plot_path}")

# 5. Raw Distribution with Best-Fit Distribution
plt.figure(figsize=(12, 8))

# Plot histogram
n, bins, patches = plt.hist(frequency, bins=100, density=True, alpha=0.7, color=sns_color)

# Find best distribution fit
distributions = [
    {'name': 'Normal', 'distribution': norm, 'color': 'red'},
    {'name': 'Student\'s t', 'distribution': t, 'color': 'green'},
    {'name': 'Skew Normal', 'distribution': skewnorm, 'color': 'purple'}
]

best_aic = np.inf
best_dist = None
best_params = None
best_name = ''
best_color = ''

# Find the best distribution based on AIC criterion
for dist_info in distributions:
    dist = dist_info['distribution']
    try:
        # Fit distribution to data
        params = dist.fit(frequency)
        
        # Calculate AIC
        log_likelihood = np.sum(dist.logpdf(frequency, *params))
        k = len(params)
        aic = 2 * k - 2 * log_likelihood
        
        if aic < best_aic:
            best_aic = aic
            best_dist = dist
            best_params = params
            best_name = dist_info['name']
            best_color = dist_info['color']
    except:
        continue

# Plot the best fit distribution
if best_dist is not None:
    x = np.linspace(min(frequency), max(frequency), 1000)
    y = best_dist.pdf(x, *best_params)
    plt.plot(x, y, color=best_color, linewidth=2, label=f'Best Fit: {best_name}')

plt.title('Raw Distribution with Best-Fit Curve', fontweight='bold')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Probability Density')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
best_fit_plot_path = os.path.join(output_dir, "monte_carlo_best_fit_distribution.png")
plt.savefig(best_fit_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Best-Fit Distribution Plot saved to: {best_fit_plot_path}")

# 6. Raw Distribution with Coverage Interval (no normal curve)
plt.figure(figsize=(12, 8))

# Plot histogram
n, bins, patches = plt.hist(frequency, bins=100, density=True, alpha=0.7, color=sns_color)

# Add vertical lines for the coverage interval boundaries
plt.axvline(x=ci_95_lower, color='green', linestyle='--', alpha=0.8, label='95% Coverage Interval')
plt.axvline(x=ci_95_upper, color='green', linestyle='--', alpha=0.8)

# Highlight the area between the coverage interval lines
for i, patch in enumerate(patches):
    bin_center = (bins[i] + bins[i+1]) / 2
    if ci_95_lower <= bin_center <= ci_95_upper:
        patch.set_facecolor('green')
        patch.set_alpha(0.6)

# Add text labels directly on the plot for better readability
# Get the current y-axis limits to position the text appropriately
plt.ylim()  # This forces matplotlib to calculate the y-limits
ymin, ymax = plt.ylim()

# Calculate offsets from base frequency for clearer display
lower_offset = ci_95_lower - fm
upper_offset = ci_95_upper - fm

# Add text labels for the lower and upper bounds inside the plot with offset values
plt.text(ci_95_lower + (ci_95_upper - ci_95_lower)*0.05, ymax*0.85, 
         f'Offset: {lower_offset:.6f} Hz', 
         fontsize=12, color='darkgreen', fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='green', boxstyle='round,pad=0.5'))

plt.text(ci_95_upper - (ci_95_upper - ci_95_lower)*0.25, ymax*0.85, 
         f'Offset: {upper_offset:.6f} Hz', 
         fontsize=12, color='darkgreen', fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='green', boxstyle='round,pad=0.5'))

plt.title('Raw Distribution with 95% Coverage Interval', fontweight='bold')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Probability Density')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
raw_coverage_plot_path = os.path.join(output_dir, "monte_carlo_raw_distribution_95coverage.png")
plt.savefig(raw_coverage_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Raw Distribution with 95% Coverage saved to: {raw_coverage_plot_path}")

# Create a summary text file with key results
summary_path = os.path.join(output_dir, "monte_carlo_summary.txt")
with open(summary_path, 'w') as f:
    f.write("===== Monte Carlo Simulation Results =====\n\n")
    f.write(f"Number of iterations: {iterations}\n")
    f.write(f"Base frequency: {fm} Hz\n\n")
    
    f.write(f"Mean Frequency: {mean:.6f} Hz\n")
    f.write(f"Standard Deviation: {std_dev:.6f} Hz\n")
    f.write(f"Skewness: {skewness:.6f}\n")
    f.write(f"Kurtosis: {kurtosis:.6f}\n\n")
    
    f.write("===== Coverage Intervals =====\n\n")
    f.write(f"95% Coverage Interval: [{ci_95_lower:.6f}, {ci_95_upper:.6f}] Hz\n")
    f.write(f"95% Coverage Interval Offsets: [{lower_offset:.6f}, {upper_offset:.6f}] Hz\n")
    f.write(f"95% Coverage Uncertainty: {uncertainty_95:.6f} Hz\n")
    f.write(f"95% Result: {mean:.6f} +/- {uncertainty_95:.6f} Hz\n\n")
    
    f.write(f"99% Coverage Interval: [{ci_99_lower:.6f}, {ci_99_upper:.6f}] Hz\n")
    f.write(f"99% Coverage Uncertainty: {uncertainty_99:.6f} Hz\n")
    f.write(f"99% Result: {mean:.6f} +/- {uncertainty_99:.6f} Hz\n\n")
    
    f.write("===== GUM Approach =====\n\n")
    f.write(f"Unexpanded Uncertainty (k=1): {unexpanded_uncertainty:.6f} Hz\n")
    f.write(f"Expanded Uncertainty (k=2): {expanded_uncertainty_k2:.6f} Hz\n\n")
    
    f.write("===== Uncertainty Contributions =====\n\n")
    f.write("{:<25} {:>15}\n".format("Contributor", "Percentage (%)"))
    f.write("-" * 42 + "\n")
    for contributor, percentage in contributors:
        f.write("{:<25} {:>15.2f}\n".format(contributor, percentage))

print(f"Summary text file saved to: {summary_path}")
