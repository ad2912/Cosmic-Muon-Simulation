# This script analyzes the angular dependence of ADC response in HGCAL layers using cosmic muon simulation.
# It reconstructs the muon angle from cluster positions in 3 layers, bins events by incidence angle,
# aggregates ADC values from HGCAL digis for selected layers, fits Landau distributions to extract MPVs,
# and plots MPV vs. angle (in degrees) for studying energy response variation with incidence angle.
#
# Input: GENSIMDIGIRECO ROOT file with reco clusters and digis
# Output: One plot per selected layer showing MPV vs. angle (degrees)

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
selected_layers = [21]
max_events = 20000  # Limit number of events
angle_bins = list(range(0, 60, 2))  # Bins of 2 degrees

# ROOT Chains
events = TChain("Events")
events.Add(filename)

# Storage for binned ADC data and histograms
adc_data_binned = {layer: {i: [] for i in range(len(angle_bins) - 1)} for layer in selected_layers}
resolution_histograms = {layer: {} for layer in selected_layers}

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
                adc_data_binned[layer_num][angle_bin].append(adc_values)

# **Step 3: Create and Fit Histograms for Each Bin**
pointsX = array.array("f", [(angle_bins[i] + angle_bins[i + 1]) / 2 for i in range(len(angle_bins) - 1)])
pointsY_mpv = {layer: [] for layer in selected_layers}
pointsY_err = {layer: [] for layer in selected_layers}

for layer_num in selected_layers:
    for angle_bin, adc_values in adc_data_binned[layer_num].items():
        if len(adc_values) > 0:
            hist_name = f"Resolution_Layer{layer_num}_Bin{angle_bin}"
            hist = TH1F(hist_name, f"Layer {layer_num}, Bin {angle_bins[angle_bin]}-{angle_bins[angle_bin + 1]}", 50, 0, 50)

            for value in adc_values:
                hist.Fill(value)

            resolution_histograms[layer_num][angle_bin] = hist
            canvas = TCanvas(hist_name, hist_name, 800, 600)
            hist.Draw()

            fit_result = hist.Fit("landau", "SQ")  # Fit Landau
            if fit_result.Status() == 0:
                fit_params = hist.GetFunction("landau")
                mpv = fit_params.GetParameter(1)
                mpv_error = fit_params.GetParError(1)

                pointsY_mpv[layer_num].append(mpv)
                pointsY_err[layer_num].append(mpv_error)
                print(f"Layer {layer_num}, Bin {angle_bins[angle_bin]}-{angle_bins[angle_bin + 1]}: MPV = {mpv:.2f} ± {mpv_error:.2f}")
            else:
                print(f"WARNING: Fit failed for Layer {layer_num}, Bin {angle_bins[angle_bin]}-{angle_bins[angle_bin + 1]}")

#            canvas.SaveAs(f"Resolution_Layer{layer_num}_Bin{angle_bin}.pdf")
            canvas.Close()
            del hist  # Free memory
# **Step 4: Plot MPV vs. Angle**
valid_pointsX = []
valid_pointsY = []
valid_pointsY_err = []

for i, mpv in enumerate(pointsY_mpv[layer_num]):
    if mpv is not None:  # Ensure valid MPV
        valid_pointsX.append((angle_bins[i] + angle_bins[i + 1]) / 2)  # Use corresponding angle midpoint
        valid_pointsY.append(mpv)
        valid_pointsY_err.append(pointsY_err[layer_num][i])  # Keep the error value

# Convert to ROOT arrays
pointsX = array.array("f", valid_pointsX)
pointsY = array.array("f", valid_pointsY)
pointsY_err = array.array("f", valid_pointsY_err)

if len(pointsX) == len(pointsY):  # Ensure lengths match
    graph = TGraphErrors(len(pointsX), pointsX, pointsY, array.array("f", [0]*len(pointsX)), pointsY_err)
    graph.SetMarkerColor(2)
    graph.SetMarkerStyle(21)
    graph.SetMarkerSize(1)
    graph.SetLineWidth(2)

    graph.SetTitle(f"MPV of ADC vs. Angle (Layer {layer_num})")
    graph.GetXaxis().SetTitle("Angle [degrees]")
    graph.GetYaxis().SetTitle("MPV of ADC counts")

    canvas = TCanvas(f"MPV_vs_Angle_Layer_{layer_num}", f"MPV vs. Angle Layer {layer_num}", 800, 600)
    canvas.cd()
    graph.Draw("APL")

    save_path = f"MPV_vs_Angle_Layer_{layer_num}.pdf"
    canvas.SaveAs(save_path)

    print(f"Final MPV vs. Angle plot saved as {save_path}")
    canvas.Close()
else:
    print(f"Error: Data mismatch in Layer {layer_num} - pointsX: {len(pointsX)}, pointsY: {len(pointsY)}")
