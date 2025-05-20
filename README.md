# Monte Carlo Simulation for 50 MHz Frequency Measurement Uncertainty Analysis

This repository contains Python code for performing large-scale Monte Carlo simulations to analyze the uncertainty in 50 MHz frequency measurements, as described in the research paper "Comprehensive Gate-Time-Resolved Uncertainty Analysis of 50 MHz Frequency Measurements via Python Based Large-Scale Monte Carlo and GUM Methods".

## Overview

The Monte Carlo simulation implemented in this code models the uncertainty in frequency measurements by considering various uncertainty components:

- Standard uncertainty
- Random uncertainty
- Display resolution
- Time base accuracy
- Systematic uncertainty

The simulation generates 400,000 iterations to provide a robust statistical analysis of the frequency measurement uncertainty.

## Features

- **Large-scale Monte Carlo simulation** with 400,000 iterations
- **Comprehensive uncertainty analysis** including:
  - Mean frequency calculation
  - Standard deviation
  - Skewness and kurtosis analysis
  - 95% and 99% coverage intervals
  - GUM (Guide to the Expression of Uncertainty in Measurement) approach comparison
  - Uncertainty contribution analysis from different components
- **Professional visualization** with multiple plot types:
  - Q-Q plot for normality assessment
  - Raw distribution histograms
  - Distribution with normal curve overlay
  - Distribution with 95% coverage interval highlighting
  - Best-fit distribution analysis
  - Raw distribution with coverage interval
- **Data export** capabilities:
  - Excel export of all simulation data
  - Comprehensive summary text file
  - High-resolution PNG plots

## Requirements

- Python 3.8+
- NumPy ≥1.26.0
- Matplotlib ≥3.10.0
- SciPy ≥1.15.0
- Pandas ≥2.2.0
- OpenPyXL ≥3.1.2 (for Excel export)

## Usage

1. Ensure all dependencies are installed:
   ```
   pip install -r requirements.txt
   ```

2. Run the Monte Carlo simulation:
   ```
   python monte_carlo_frequency_uncertainty.py
   ```

3. The script will create a timestamped output directory containing:
   - Excel file with all frequency data
   - Multiple visualization plots in PNG format
   - A comprehensive summary text file

## Examples

Sample outputs including plots and summary files can be found in the `/examples` folder for reference and reproducibility.


## Model Description

The frequency measurement model used in this simulation is:

```
f = fm(1 + display_resolution + systematic_uncertainty) + standard_uncertainty + random_uncertainty + time_base_accuracy
```

Where:
- `fm` is the base frequency (50 MHz)
- Each uncertainty component is modeled with appropriate probability distributions

## Output

The simulation provides:
- Statistical analysis of the frequency distribution
- Coverage intervals at 95% and 99% confidence levels
- Comparison with GUM approach
- Detailed breakdown of uncertainty contributions
- Visual representations of the frequency distribution

## Citation

If you use this code in your research, please cite the original research paper:
"Comprehensive Gate-Time-Resolved Uncertainty Analysis of 50 MHz Frequency Measurements via Python Based Large-Scale Monte Carlo and GUM Methods"

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
