%% fig3.m - The code to plot figure 3 of "An active electronic, high-density epidural paddle array for chronic spinal cord neuromodulation"
% Author: Samuel Parker
% Last Modified: 27-Sep-2024

clear all; close all; clc
%% Panel A

load(strcat("data", filesep, "fig3_panelA.mat"))
figure()
tcl = tiledlayout(2, 1);

nexttile
area(fig3_panelA.emg_trace.time_ms, abs(fig3_panelA.emg_trace.emg_mV), "FaceColor", "#bff4ff", "EdgeColor", "#00d3ff"); hold on
plot(fig3_panelA.emg_trace.time_ms, abs(fig3_panelA.emg_trace.emg_mV), "Color", "#00d3ff");
plot(fig3_panelA.emg_trace.time_ms, fig3_panelA.emg_trace.emg_mV, "Color", "#000000");
xticks(0:200:400)
xlim([-50, 450])
xlabel("Time (s)")
ylabel("EMG (mV)")
legend("rAUC", "Rectified", "Raw", "Location", "NorthEast")

nexttile
musc_names = ["RightEDL", "RightBF", "RightGas", "LeftEDL", "LeftBF", "LeftGas"];
colors = ["#06D6A0", "#B04CB9", "#073B4C", "#EF476F", "#FFD166", "#FD833D"];
for i = 1:length(musc_names)
    errorbar(fig3_panelA.rec_curve.StimAmp_uA, fig3_panelA.rec_curve.(strcat(musc_names(i), "_norm_mean")), fig3_panelA.rec_curve.(strcat(musc_names(i), "_norm_std")), "Color", colors(i)); hold on
end
xticks(500:500:1500)
xlim([450, 1550])
xlabel("EES amp (uA)")
ylabel("rAUC (a.u.)")

%% Panel B - this section saves SVGs to the current directory instead of plotting

load(strcat("data", filesep, "fig3_panelB.mat"))

cmap = abyss(256);
amps_normalized = (fig3_panelB{:, 2:end} - 250) / (1500); 
cmap_amp_vals = round(amps_normalized .* 256) + 1;  
musc_names = ["RightEDL", "RightBF", "RightGas", "LeftEDL", "LeftBF", "LeftGas"];

for musc = 1:6
    clear hex

    this_elecs = unique(fig3_panelB.stimContact);
    this_elecs(isnan(cmap_amp_vals(:, musc))) = [];
    this_idxs = cmap_amp_vals(:, musc);
    this_idxs(isnan(cmap_amp_vals(:, musc))) = [];
    %this_idxs(isnan(cmap_amp_vals(:, musc))) = 256;

    rgb = round(cmap(this_idxs, :) .* 255);
    hex(:,2:7) = reshape(sprintf('%02X',rgb.'),6,[]).'; 
    hex(:,1) = '#';
    hex = string(hex);

    out = plotHD64(this_elecs, hex);
    filename = sprintf("minAmpForThreshold_%s.svg", musc_names(musc));
    fprintf("Saving %s...\n", filename);
    fid = fopen(filename, "w+");
    fprintf(fid, "%s", out);
    fclose(fid);
end
disp("Done!")

%% Panel C

load(strcat("data", filesep, "fig3_panelC.mat"))
colors = ["#4d873b", "#506887", "#fff275", "#ff8c42", "#b3c6a1", "#ff3c38", "#a23e48", "#6699cc"];

figure()
gscatter(fig3_panelC.UMAP1, fig3_panelC.UMAP2, fig3_panelC.clusterID, hex2rgb(colors));
legend(string(strsplit(num2str(0:7), " ")), "NumColumns",2, "Location", "northeast");
xticks([])
yticks([])
xlabel("UMAP1")
ylabel("UMAP2")

%% Panel D - this section saves SVGs to the current directory instead of plotting

load(strcat("data", filesep, "fig3_panelD.mat"))
colors = ["#4d873b", "#506887", "#fff275", "#ff8c42", "#b3c6a1", "#ff3c38", "#a23e48", "#6699cc"];

out = plotHD64(fig3_panelD.ContactID, colors(fig3_panelD.ClusterID+1));
disp("Saving fig3_panelD.svg...")
fid = fopen("fig3_panelD.svg", "w+");
fprintf(fid, "%s", out);
fclose(fid);
disp("Done!")

%% functions

function outColor = hex2rgb(string_vector)
    outColor = nan(length(string_vector), 3);
    for i = 1:length(string_vector)
        hex = char(string_vector(i));
        if hex(1) == "#"
            hex(1) = [];
        end
        r = hex2dec(hex(1:2)) ./255;
        g = hex2dec(hex(3:4)) ./255;
        b = hex2dec(hex(5:6)) ./255;
        outColor(i, :) = [r, g, b];
    end
end