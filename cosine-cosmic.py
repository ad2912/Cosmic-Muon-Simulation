# This script analyzes energy deposition in selected HGCAL layers using simulated cosmic muons.
# It reconstructs cluster hit positions, estimates incidence angles from track slopes,
# bins events by angle, fits ADC distributions with Landau functions,
# and plots the MPV vs. 1/cos(theta) to study angular dependence of energy response.

# Input: GENSIMDIGIRECO ROOT file with reco clusters and digis
# Output: One plot per selected layer showing MPV vs. 1/cos(theta)


from ROOT import TChain, TCanvas, TGraphErrors, TH1F, gROOT, TF1
import numpy as np
import math
import array

# Helper function to calculate angle from slope
def calculate_angle(slope):
    return math.degrees(math.atan(slope))

# Extract layer number from DetId
def layer(id):
    return (int(id) >> 17) & 0x1F

# Bin angles into specified ranges
def get_angle_bin(angle, bins):
    for i in range(len(bins) - 1):
        if bins[i] <= angle < bins[i + 1]:
            return i
    return None  # If out of range

# Define angle bins and selected layers
angle_bins = list(range(0, 60, 2))  # Bins of 2 degrees
selected_layers = [9]#, 11, 13, 15, 17, 19]

# File details
filename = "gensimdigireco_muon5Cosmic-Muon-Distribution-20kevents.root"
max_events = 20000

# Storage for MPV results
pointsX = []
pointsY_mpv = {layer: [] for layer in selected_layers}
pointsY_err = {layer: [] for layer in selected_layers}
resolution_histograms = {layer: {} for layer in selected_layers}

# **Process events and calculate angles**
events = TChain("Events")
events.Add(filename)

for i, event in enumerate(events):
    if i >= max_events:
        break
    recHits = getattr(event, "recoCaloClusters_hgcalLayerClustersHSci__GENSIMDIGIRECO").product()
    z_hits = {layer: [] for layer in selected_layers}
    
    for hit in recHits:
        x, z = hit.x(), hit.z()
        for target_z in selected_layers:
            if abs(z - target_z) <= 1.0:
                z_hits[target_z].append(x)
                break
    
    if all(len(z_hits[layer]) > 0 for layer in selected_layers):
        sorted_z = sorted(z_hits.keys())
        x_vals = np.array([sum(z_hits[z]) / len(z_hits[z]) for z in sorted_z], dtype='d')
        z_vals = np.array(sorted_z, dtype='d')
        
        graph = TGraphErrors(len(z_vals), z_vals, x_vals, np.zeros(len(z_vals)), np.zeros(len(z_vals)))
        fit_result = graph.Fit("pol1", "QS")
        slope = fit_result.Parameter(1)
        angle = calculate_angle(slope)
        
        angle_bin = get_angle_bin(angle, angle_bins)
        if angle_bin is None:
            continue
        
        pointsX.append(1 / np.cos(math.radians(angle)))
        
        digis = getattr(event, "DetIdHGCSampleHGCDataFramesSorted_mix_HGCDigisHEback_GENSIMDIGIRECO").product()
        for digi in digis:
            layer_num = layer(digi.id())
            adc_values = sum(sample.data() for sample in digi)
            
            if layer_num in selected_layers:
                if angle_bin not in resolution_histograms[layer_num]:
                    resolution_histograms[layer_num][angle_bin] = TH1F(f"Resolution_Layer{layer_num}_Bin{angle_bin}",
                                                                       f"Layer {layer_num}, Bin {angle_bins[angle_bin]}-{angle_bins[angle_bin + 1]}",
                                                                       50, 0, 50)
                resolution_histograms[layer_num][angle_bin].Fill(adc_values)

# **Fit Landau distributions and extract MPV values**
for layer_num in selected_layers:
    for angle_bin in resolution_histograms[layer_num]:
        hist = resolution_histograms[layer_num][angle_bin]
        
        if hist.GetEntries() > 0:
            fit_result = hist.Fit("landau", "SQ")
            fit_params = hist.GetFunction("landau")
            mpv = fit_params.GetParameter(1)
            mpv_error = fit_params.GetParError(1)
            pointsY_mpv[layer_num].append(mpv)
            pointsY_err[layer_num].append(mpv_error)
        else:
            pointsY_mpv[layer_num].append(0)
            pointsY_err[layer_num].append(0)

# **Plot MPV vs. 1/cos(theta) and fit with linear function**
for layer_num in selected_layers:
  if len(pointsX) != len(pointsY_mpv[layer_num]):
    print(f"Warning: Mismatch in data sizes for layer {layer_num}")

    x_values = array.array("f", list(map(float, pointsX)))
    y_values = array.array("f", list(map(float, pointsY_mpv[layer_num])))
    y_errors = array.array("f", list(map(float, pointsY_err[layer_num])))

    
    graph = TGraphErrors(len(x_values), x_values, y_values, array.array("f", [0]*len(x_values)), y_errors)
    fit_function = TF1(f"fit_layer_{layer_num}", "[0]*x", min(x_values), max(x_values))
    graph.Fit(fit_function, "Q")  # Fit quietly
    
    k_value = fit_function.GetParameter(0)
    k_error = fit_function.GetParError(0)
    print(f"Layer {layer_num}: k = {k_value:.3f} ± {k_error:.3f}")
    
    graph.SetTitle(f"MPV vs. 1/cos(theta) (Layer {layer_num})")
    graph.GetXaxis().SetTitle("1 / cos(theta)")
    graph.GetYaxis().SetTitle("MPV of ADC Counts")
    
    canvas = TCanvas(f"Layer_{layer_num}_MPV", f"Layer {layer_num} MPV", 800, 600)
    graph.Draw("AP")
    canvas.SaveAs(f"ADC_vs_CosineLaw_Layer_{layer_num}.pdf")
