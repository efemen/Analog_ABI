from single_force_edgewise import run_single_force

# Define an array of RPM values to test
rpm_array = [3500, 3000, 2500, 2000]  # Example RPM values; adjust as needed

# Loop through each RPM and run the experiment sequentially
for rpm in rpm_array:
    print(f"Starting experiment for RPM: {rpm}")
    run_single_force(RPM=rpm)
    print(f"Finished experiment for RPM: {rpm}")

print("All experiments completed.")
