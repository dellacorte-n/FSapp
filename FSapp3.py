import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import matplotlib.pyplot as plt
import io

# Configuration / Parameters
rng = np.random.default_rng(42)
NUM_SAMPLES = 100_000
DEFAULT_SIZE_BAND = 0.80
STRESS_BINS_MPA = [(0, 3), (3, 6), (6, 9), (9, 12), (12, np.inf)]
STRESS_BIN_LABELS = ["0–3", "3–6", "6–9", "9–12", "12+"] 

# Utility Functions
def calculate_stress_mpa_from_mN(force_mN, size_um):
    force_N = np.asarray(force_mN, dtype=float) * 1e-3
    size_m = np.asarray(size_um, dtype=float) * 1e-6
    area_m2 = np.pi * (size_m / 2.0) ** 2
    stress_pa = np.divide(force_N, area_m2, out=np.zeros_like(force_N, dtype=float), where=area_m2 > 0)
    return stress_pa / 1e6 

def sample_truncnorm(mean, sd, low, high, size, rng):
    a = -np.inf if low is None else (low - mean) / sd 
    b = np.inf if high is None else (high - mean) / sd 
    return stats.truncnorm.rvs(a=a, b=b, loc=mean, scale=sd, size=size, random_state=rng) 

def size_band_to_bounds(mean_size_um, std_size_um, band):
    tail = (1.0 - band) / 2.0 
    low_q = stats.norm.ppf(tail, loc=mean_size_um, scale=std_size_um) 
    high_q = stats.norm.ppf(1.0 - tail, loc=mean_size_um, scale=std_size_um) 
    return float(max(low_q, 1e-6)), float(high_q)

def stress_bin_percentages(stress_mpa, bins, labels):
    pct = []
    for (lo, hi) in bins:
        if np.isfinite(hi):
            mask = (stress_mpa >= lo) & (stress_mpa < hi)
        else:
            mask = (stress_mpa >= lo)
        pct.append(100.0 * np.mean(mask))
    return dict(zip(labels, pct))

def empirical_mode_hist(x, bins=256):
    hist, edges = np.histogram(x, bins=bins)
    idx = np.argmax(hist)
    return 0.5 * (edges[idx] + edges[idx + 1])

def simulate_combined_sample(all_samples, num_samples=NUM_SAMPLES):
    combined_stress_samples = []
    statistics_data = []
    bin_percentages_data = []

    for sample in all_samples:
        force_data, mean_size, std_dev_size, sample_name = sample

        # Convert to numeric and handle errors
        force_values = pd.to_numeric(force_data, errors='coerce').dropna()
        
        # Check if force_values is empty after conversion
        if force_values.empty:
            st.error(f"The selected force data column for {sample_name} contains no valid numeric data.")
            continue  # Skip this sample

        mean_force = np.mean(force_values)
        std_dev_force = np.std(force_values)

        if std_dev_force == 0:
            st.error(f"The standard deviation of the force data for {sample_name} is zero. Please provide varied data.")
            continue

        size_low_um, size_high_um = size_band_to_bounds(mean_size, std_dev_size, band=DEFAULT_SIZE_BAND)

        if std_dev_size == 0:
            st.error(f"The standard deviation for size cannot be zero for {sample_name}.")
            continue

        # Monte Carlo samples
        force_mN = sample_truncnorm(mean_force, std_dev_force, low=0.0, high=None, size=num_samples, rng=rng)
        size_um = sample_truncnorm(mean_size, std_dev_size, low=size_low_um, high=size_high_um, size=num_samples, rng=rng)
        stress_mpa = calculate_stress_mpa_from_mN(force_mN, size_um)

        combined_stress_samples.extend(stress_mpa)

        # Collect statistics for each sample
        mean_stress = np.mean(stress_mpa)
        std_dev_stress = np.std(stress_mpa)
        sem_stress = std_dev_stress / np.sqrt(len(stress_mpa))  # Standard Error of the Mean
        cv_stress = (std_dev_stress / mean_stress) * 100 if mean_stress != 0 else 0  # Coefficient of Variation

        statistics_data.append({
            "Sample Name": sample_name,
            "Mean Rupture Force (mN)": round(mean_force, 2),
            "Rupture Force StDev (mN)": round(std_dev_force, 2),
            "Mean Fracture Strength (MPa)": round(mean_stress, 2),
            "Fracture Strength StDev (MPa)": round(std_dev_stress, 2),
            "Median Fracture Strength (MPa)": round(np.median(stress_mpa), 2),
            "Empirical Fracture Strength Mode (MPa)": round(empirical_mode_hist(stress_mpa), 2)
        })

        # Percentages in bins
        pct_bins = stress_bin_percentages(stress_mpa, STRESS_BINS_MPA, STRESS_BIN_LABELS)
        bin_percentages_data.append({**{"Sample Name": sample_name}, **{label: int(value) for label, value in pct_bins.items()}})

    # Convert combined stress samples to numpy array for statistics
    combined_stress_samples = np.array(combined_stress_samples)

    # Overall statistics for combined samples
    overall_mean_stress = float(np.mean(combined_stress_samples))
    overall_std_stress = float(np.std(combined_stress_samples))
    overall_median_stress = float(np.median(combined_stress_samples))
    overall_mode_empirical = empirical_mode_hist(combined_stress_samples, bins=256)
    
    overall_bin_percentages = stress_bin_percentages(combined_stress_samples, STRESS_BINS_MPA, STRESS_BIN_LABELS)

    results = {
        "overall_mean_stress": overall_mean_stress,
        "overall_std_stress": overall_std_stress,
        "overall_median_stress": overall_median_stress,
        "overall_mode_empirical": overall_mode_empirical,
        "overall_bin_percentages": overall_bin_percentages,
        "combined_stress_samples": combined_stress_samples,
        "statistics_data": statistics_data,
        "bin_percentages_data": bin_percentages_data,
    }
    return results

def main():
    st.title("SandCaps Volume Weighted Fracture Strength Calculator")

    # Input for number of samples
    num_samples = st.number_input("Number of Samples", min_value=1, value=1, step=1)

    all_samples = []

    for i in range(num_samples):
        st.subheader(f"Sample {i + 1}")

        # Sample name input
        sample_name = st.text_input(f"Enter a Name for Sample {i + 1}", value=f"Sample_{i + 1}", key=f"sample_name_{i}")

        # Manual input for mean size and standard deviation
        mean_size = st.number_input(f"Enter the capsule Mean Size in µm from Accusizer for {sample_name}", value=0.0, key=f"mean_size_{i}")
        std_dev_size = st.number_input(f"Enter the capsule Size StDev in µm from Accusizer for {sample_name}", value=0.0, key=f"std_dev_size_{i}")

        # File upload for force and size data
        force_size_file = st.file_uploader(f"Upload the Excel File containing Rupture Force measurements for {sample_name}", type=["xlsx"], key=f"force_size_file_{i}")

        if force_size_file:
            # Read the uploaded Excel file up to the specified row
            force_size_data = pd.read_excel(force_size_file)

            # Display column selection for force and size
            st.write("Excel file preview:")
            # Set the height to show more rows
            st.dataframe(force_size_data, height=300)  # Adjust height to fit more rows

            # Select columns for force and size
            force_column = st.selectbox(f"Select the column containing the Rupture Force data for {sample_name}", options=force_size_data.columns.tolist())
            size_column = st.selectbox(f"Select the column containing the measured Diameter data for {sample_name}", options=force_size_data.columns.tolist())

            # Store the data for cumulative simulation
            all_samples.append((force_size_data[force_column], mean_size, std_dev_size, sample_name))


        # Input for the maximum row number to read
        max_row = st.number_input(f"Enter the maximum row number to read for {sample_name} - keep 32 for Bham files", min_value=1, value=32, step=1, key=f"max_row_{i}")

    # Run Cumulative Simulation
    if st.button("Run Simulation"):
        if all_samples:
            results = simulate_combined_sample(all_samples)

            # Create DataFrames for statistics and bin percentages
            stats_df = pd.DataFrame(results['statistics_data'])
            bins_df = pd.DataFrame(results['bin_percentages_data']).fillna(0)  # Fill NaN values with 0 for missing bin data
            
            # Display the statistics summary table
            st.subheader("Statistics summary")
            st.dataframe(stats_df)

            # Display the bin percentages table with color coding
            st.subheader("Percentage of Capsules in Fracture Strength Bins")
            bins_df_style = bins_df.style.background_gradient(cmap='YlGn', axis=None)
            st.dataframe(bins_df_style)

            # Create a downloadable CSV of the results
            output = io.StringIO()
            stats_df.to_csv(output, index=False)
            bins_df.to_csv(output, index=False)
            st.download_button(
                label="Download Statistics and Bins",
                data=output.getvalue(),
                file_name="simulation_results.csv",
                mime="text/csv"
            )

            # Combined graph of stress distributions for all samples
            fig, ax = plt.subplots()
            for sample in all_samples:
                force_data, mean_size, std_dev_size, sample_name = sample
                
                # Simulate using the calculated stress data
                force_values = pd.to_numeric(force_data, errors='coerce').dropna()
                
                if not force_values.empty:  # Proceed only if there are valid values
                    # Calculate simulated stress data
                    mean_force = np.mean(force_values)
                    std_dev_force = np.std(force_values)
                    size_low_um, size_high_um = size_band_to_bounds(mean_size, std_dev_size, band=DEFAULT_SIZE_BAND)
                    simulated_force_mN = sample_truncnorm(mean_force, std_dev_force, low=0.0, high=None, size=NUM_SAMPLES, rng=rng)
                    simulated_size_um = sample_truncnorm(mean_size, std_dev_size, low=size_low_um, high=size_high_um, size=NUM_SAMPLES, rng=rng)
                    simulated_stress_mpa = calculate_stress_mpa_from_mN(simulated_force_mN, simulated_size_um)

                    # Calculate KDE for the simulated stress data
                    kde = stats.gaussian_kde(simulated_stress_mpa)
                    x = np.linspace(min(simulated_stress_mpa), max(simulated_stress_mpa), 100)
                    y = kde(x)

                    # Plot the KDE
                    ax.plot(x, y, label=f'{sample_name}')

            # Set the x-axis limit to 30 MPa
            ax.set_xlim(0, 30)

            ax.set_title('Volume Weighted Fracture Strength Distribution for All Samples')
            ax.set_xlabel('Volume Weighted Fracture Strength (MPa)')
            ax.set_ylabel('Density')
            ax.legend()
            st.pyplot(fig)

            # Scatter plot for bin percentages with dashed lines
            fig_bins, ax_bins = plt.subplots()

            # Plot individual sample bin percentages
            for bin_data in results['bin_percentages_data']:
                sample_name = bin_data["Sample Name"]
                bin_values = [bin_data[label] for label in STRESS_BIN_LABELS]
                
                # Scatter plot for individual sample percentages
                ax_bins.scatter(STRESS_BIN_LABELS, bin_values, label=f'{sample_name}', s=50)
                # Add dashed line connecting the points for individual samples
                ax_bins.plot(STRESS_BIN_LABELS, bin_values, linestyle='--')

            ax_bins.set_title('Volume Weighted Fracture Strength Bin Percentages for Each Sample')
            ax_bins.set_xlabel('Volume Weighted Fracture Strength Bins')
            ax_bins.set_ylabel('Percentage of Capsules (%)')
            ax_bins.legend()
            st.pyplot(fig_bins)

if __name__ == "__main__":
    main()