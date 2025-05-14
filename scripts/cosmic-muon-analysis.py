# This script reconstructs muon angles using HGCAL cluster positions and correlates those angles
# with ADC signal distributions across selected detector layers. It fits a line to x vs. z positions
# of clusters to determine the incidence angle, then extracts ADC counts from digis in specified layers.
# For each reconstructed angle, ADC histograms are filled and fitted with Landau functions to extract
# the most probable value (MPV), which is plotted against angle to study angular dependence of energy deposition.
#
# Input: GENSIMDIGIRECO ROOT file with cluster and digi collections
# Output:
#   - Reconstructed angle distribution histogram
#   - ADC resolution histograms per angle
#   - MPV vs. angle plots with error bars for each selected layer

from ROOT import TChain, TCanvas, TGraphErrors, TH1F, TLatex, gROOT
import numpy as np
import math
import array

# Helper function to calculate angle from slope
def calculate_angle(slope):
    angle_rad = math.atan(slope)  # Convert slope to radians
    angle_deg = math.degrees(angle_rad)  # Convert radians to degrees
    return angle_deg

# Function to calculate average x values and constant error for each layer
def average_hits(z_hits):
    averaged_hits = {}
    errors = {}
    for z, coords in z_hits.items():
        avg_x = sum(coords) / len(coords)  # Mean x value for each z layer
        error = 3.74 / math.sqrt(12)  # Fixed error
        averaged_hits[z] = avg_x
        errors[z] = error
    return averaged_hits, errors

# Function to map detector ID to layer number
def layer(id):
    kHGCalLayerOffset = 17
    kHGCalLayerMask = 0x1F
    return (int(id) >> kHGCalLayerOffset) & kHGCalLayerMask

# Process a single file to analyze reconstructed angles and ADC counts
def process_file(filename, target_layers, selected_layers, tolerance=1.0):
    events = TChain("Events")
    events.Add(filename)

    angular_distribution_hist = TH1F("angular_distribution", "Angular Distribution of Reconstructed Muons", 100, 0, 90)

    resolution_histograms = {layer: {} for layer in selected_layers}
    pointsX = []  # Reconstructed angles
    pointsY_mpv = {layer: [] for layer in selected_layers}
    pointsY_err = {layer: [] for layer in selected_layers}

    for event in events:
        recHits = getattr(event, "recoCaloClusters_hgcalLayerClustersHSci__GENSIMDIGIRECO").product()
        digis = getattr(event, "DetIdHGCSampleHGCDataFramesSorted_mix_HGCDigisHEback_GENSIMDIGIRECO").product()

        # Step 1: Reconstruct angles
        z_hits = {layer: [] for layer in target_layers}
        for hit in recHits:
            x, z = hit.x(), hit.z()
            for target_z in target_layers:
                if abs(z - target_z) <= tolerance:
                    z_hits[target_z].append(x)
                    break

        if all(len(z_hits[layer]) > 0 for layer in target_layers):
            averaged_hits, errors = average_hits(z_hits)
            sorted_z = sorted(averaged_hits.keys())
            z_vals = np.array(sorted_z, dtype='d')  # Z positions
            x_vals = np.array([averaged_hits[z] for z in sorted_z], dtype='d')
            x_errors = np.array([errors[z] for z in sorted_z], dtype='d')

            graph = TGraphErrors(len(z_vals), z_vals, x_vals, np.zeros(len(z_vals)), x_errors)
            fit_result = graph.Fit("pol1", "QS")  # Fit line quietly and save result
            slope = fit_result.Parameter(1)  # Extract slope (dx/dz)
            reconstructed_angle = calculate_angle(slope)

            angular_distribution_hist.Fill(reconstructed_angle)
            pointsX.append(reconstructed_angle)

        # Step 2: Calculate ADC counts for the same event
        event_adc = {}
        for digi in digis:
            layer_num = layer(digi.id())
            adc_values = [sample.data() for sample in digi]  # List of ADC counts from each time sample

            if layer_num not in event_adc:
                event_adc[layer_num] = []

            event_adc[layer_num].append(sum(adc_values))

        for layer_num, adc_values in event_adc.items():
            if layer_num in selected_layers:
                if reconstructed_angle not in resolution_histograms[layer_num]:
                    resolution_histograms[layer_num][reconstructed_angle] = TH1F(
                        f"Resolution_Layer{layer_num}_Angle{reconstructed_angle}",
                        f"Resolution: Layer {layer_num}, Angle {reconstructed_angle}",
                        50, 0, 50  # Modify range as needed
                    )

                hist = resolution_histograms[layer_num][reconstructed_angle]
                for adc in adc_values:
                    hist.Fill(adc)  # Fill with sum of ADC values per event

    # Plot Angular Distribution
    canvas1 = TCanvas("canvas1", "Angular Distribution", 800, 600)
    angular_distribution_hist.GetXaxis().SetTitle("Reconstructed Angle (degrees)")
    angular_distribution_hist.GetYaxis().SetTitle("Frequency")
    angular_distribution_hist.SetLineColor(4)  # Blue
    angular_distribution_hist.SetLineWidth(2)
    angular_distribution_hist.Draw()
    canvas1.SaveAs("Reconstructed_Angular_Distribution_xz.pdf")

    # Step 3: Plot Resolution and MPV vs. Reconstructed Angle
    for layer_num in selected_layers:
        for angle, hist in resolution_histograms[layer_num].items():
            if hist.GetEntries() > 0:
                canvas2 = TCanvas(f"Resolution_Layer{layer_num}_Angle{angle}", f"Resolution Layer {layer_num}, Angle {angle}", 800, 600)
                hist.GetXaxis().SetTitle("Total ADC Counts")
                hist.GetYaxis().SetTitle("Number of Events")
                hist.Draw()

                fit_result = hist.Fit("landau", "SQ")  # Silent fit
                fit_params = hist.GetFunction("landau")

                mpv = fit_params.GetParameter(1)  # Most Probable Value
                mpv_error = fit_params.GetParError(1)  # Error on MPV

                pointsY_mpv[layer_num].append(mpv)
                pointsY_err[layer_num].append(mpv_error)

                canvas2.SaveAs(f"Resolution_Layer{layer_num}_Angle{angle}_Fit.pdf")

        # Plot MPV vs. Reconstructed Angle
        y_values = array.array("f", pointsY_mpv[layer_num])
        y_errors = array.array("f", pointsY_err[layer_num])

        graph = TGraphErrors(len(pointsX), array.array("f", pointsX), y_values, array.array("f", [0]*len(pointsX)), y_errors)
        graph.SetMarkerColor(1)
        graph.SetMarkerSize(1)
        graph.SetLineWidth(2)

        graph.SetTitle(f"MPV of ADC vs. Reconstructed Angle (Layer {layer_num})")
        graph.GetXaxis().SetTitle("Reconstructed Angle [degrees]")
        graph.GetYaxis().SetTitle("MPV of ADC")

        canvas3 = TCanvas(f"Layer_{layer_num}_MPV", f"Layer {layer_num} MPV", 800, 600)
        graph.Draw("APL")
        canvas3.SaveAs(f"MPV_vs_Angle_Layer_{layer_num}_withErrorBars.pdf")

# Configuration details
filename = "gensimdigireco_muon5Cosmic-Muon-Distribution-20kevents.root"
target_layers = [418, 463, 513]  # Z positions of layers to use
selected_layers = [9]#, 11, 13, 15, 17, 19]  # Layer numbers for ADC counts
process_file(filename, target_layers, selected_layers)
