# This script computes the normalized ADC energy resolution for cosmic muons in HGCAL across all angles.
# It reconstructs the incident angle from cluster positions, bins ADC values by angle, and compares measured
# ADC to an expected value scaled as 1/cos(θ)=k=10.995. A global histogram of (ADC_measured - ADC_expected)/ADC_expected
# is filled and plotted to assess resolution behavior over a wide angular range.
#
# Input: GENSIMDIGIRECO ROOT file with reco clusters and digis
# Output: One combined histogram showing normalized ADC resolution across all angle bins


from ROOT import TChain, TCanvas, TGraphErrors, TH1F, gROOT
import numpy as np
import math
import array

# Helper function to calculate angle from slope
def calculate_angle(slope):
    return math.degrees(math.atan(slope))

# Function to calculate average x values and errors for each layer
def average_hits(z_hits):
    return {z: sum(coords) / len(coords) for z, coords in z_hits.items()}, {z: 3.74 / math.sqrt(12) for z in z_hits}

# Extract layer number from DetId
def layer(id):
    return (int(id) >> 17) & 0x1F

# Bin angles into specified ranges
def get_angle_bin(angle, bins):
    for i in range(len(bins) - 1):
        if bins[i] <= angle < bins[i + 1]:
            return i
    return None  # If out of range

# File details
filename = "gensimdigireco_muon5Cosmic-Muon-Distribution-20kevents.root"
target_layers = [418, 463, 513]
selected_layers = [9]
max_events = 30000  # Limit number of events
angle_bins = list(range(0, 60, 5))  # Bins of 2 degrees

# ROOT Chains
events = TChain("Events")
events.Add(filename)

# Storage for binned ADC data and histograms
adc_data_binned = {layer: {i: [] for i in range(len(angle_bins) - 1)} for layer in selected_layers}
resolution_histograms = {layer: {} for layer in selected_layers}

# Create a single histogram to store resolution data for all angles
hist_resolution_all = TH1F("Resolution_AllAngles", "Normalized ADC Energy Resolution (All Angles)", 80, -3.0, 3.0)

# **Step 1: Calculate Angles and Bin Data**
event_angles = []
for i, event in enumerate(events):
    if i >= max_events:
        break  # Limit events
    recHits = getattr(event, "recoCaloClusters_hgcalLayerClustersHSci__GENSIMDIGIRECO").product()
    z_hits = {layer: [] for layer in target_layers}

    for hit in recHits:
        x, z = hit.x(), hit.z()
        for target_z in target_layers:
            if abs(z - target_z) <= 1.0:
                z_hits[target_z].append(x)
                break

    if all(len(z_hits[layer]) > 0 for layer in target_layers):
        averaged_hits, errors = average_hits(z_hits)
        sorted_z = sorted(averaged_hits.keys())
        z_vals = np.array(sorted_z, dtype='d')
        x_vals = np.array([averaged_hits[z] for z in sorted_z], dtype='d')

        graph = TGraphErrors(len(z_vals), z_vals, x_vals, np.zeros(len(z_vals)), np.zeros(len(z_vals)))
        fit_result = graph.Fit("pol1", "QS")
        slope = fit_result.Parameter(1)
        angle = calculate_angle(slope)
        event_angles.append(angle)

        # Determine angle bin
        angle_bin = get_angle_bin(angle, angle_bins)
        if angle_bin is None:
            continue  # Skip if out of range

        # **Step 2: Read ADC Counts and Aggregate by Angle Bin**
        digis = getattr(event, "DetIdHGCSampleHGCDataFramesSorted_mix_HGCDigisHEback_GENSIMDIGIRECO").product()
        for digi in digis:
            layer_num = layer(digi.id())
            adc_values = sum(sample.data() for sample in digi)  # Sum ADC values

            if layer_num in selected_layers:
                adc_data_binned[layer_num][angle_bin].append((adc_values, angle))

# **Step 3: Create and Fit Histograms for Each Bin**
for layer_num in selected_layers:
    for angle_bin, adc_values in adc_data_binned[layer_num].items():
        if len(adc_values) > 0:
            hist_name = f"Resolution_Layer{layer_num}_Bin{angle_bin}"
            hist = TH1F(hist_name, f"Layer {layer_num}, Bin {angle_bins[angle_bin]}-{angle_bins[angle_bin + 1]}", 50, 0, 50)

            for adc_measured, _ in adc_values:  # Extract only ADC measured value
                hist.Fill(adc_measured)

            resolution_histograms[layer_num][angle_bin] = hist
            canvas = TCanvas(hist_name, hist_name, 800, 600)
            hist.Draw()
            canvas.Close()
            del hist  # Free memory

# **Step 4: Merge ADC Resolution for All Angles**
for layer_num in selected_layers:
    for angle_bin in adc_data_binned[layer_num]:
        if len(adc_data_binned[layer_num][angle_bin]) > 0:
            for adc_measured, theta_reco in adc_data_binned[layer_num][angle_bin]:
                adc_predicted = 10.995 / math.cos(math.radians(theta_reco))  # Compute per event
                adc_resolution = (adc_measured - adc_predicted) / adc_predicted  # Normalize

                hist_resolution_all.Fill(adc_resolution)  # Fill the global histogram

# **Step 5: Plot and Save Merged Histogram**
canvas = TCanvas("Resolution_AllAngles", "Normalized ADC Energy Resolution (All Angles)", 800, 600)
hist_resolution_all.GetXaxis().SetTitle("(ADC Measured - ADC Predicted) / ADC Predicted")
hist_resolution_all.GetYaxis().SetTitle("Number of Events")
hist_resolution_all.Draw()

save_path = "Normalized_ADC_Energy_Resolution_AllAngles.pdf"
canvas.SaveAs(save_path)
print(f"Saved: {save_path}")

canvas.Close()
