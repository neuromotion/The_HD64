%% fig4.m - The code to plot figure 4 of "An active electronic bidirectional interface for high resolution interrogation of the spinal cord"
% Author: Samuel Parker
% Last Modified: 21-May-2024

clear all; close all; clc

%% Panels A - I

load("fig4_panelsA_I.mat")

cathodes = [3, 3, 3];
anodes = [-1, 4, 14];
elec_configs = ["C3", "C3A4", "C3A14"];
config_names = ["Mono", "Narrow Bipolar", "Wide Bipolar"];
musc_names = ["RightEDL", "RightBF", "RightGas", "LeftEDL", "LeftBF", "LeftGas"];
colors = ["#06D6A0", "#B04CB9", "#073B4C", "#EF476F", "#FFD166", "#FD833D"];

% Plot HD64 maps, save them as SVGs
for i = 1:length(elec_configs)
    fid = fopen(strcat(config_names(i), "HD64 Map.svg"), "w+");
    out = plotHD64(cathodes(i), anodes(i));
    fprintf(fid, "%s", out);
    fclose(fid);
end

figure()
% Plot mono vs narrow rAUC and Selectivity Index
tcl = tiledlayout(1, 2);
nexttile
for i = 1:length(musc_names)
    errorbar(fig4_panelsA_I.stimAmp_uA, fig4_panelsA_I.(strcat(elec_configs(1), "_", musc_names(i), "_mean_rAUC")), fig4_panelsA_I.(strcat(elec_configs(1), "_", musc_names(i), "_std_rAUC")), "--", "Color", colors(i)); hold on
    errorbar(fig4_panelsA_I.stimAmp_uA, fig4_panelsA_I.(strcat(elec_configs(2), "_", musc_names(i), "_mean_rAUC")), fig4_panelsA_I.(strcat(elec_configs(2), "_", musc_names(i), "_std_rAUC")), "-", "Color", colors(i)); hold on
end
legend_str = repmat("", 2*length(musc_names), 1); legend_str(2:2:end) = musc_names;
legend(legend_str, 'Location', "southoutside", "Orientation", "horizontal", "NumColumns", 2);
xticks(500:500:1500)
xlim([450, 1550])
xlabel("EES amp (uA)")
ylabel("rAUC (a.u.)")

nexttile
for i = 1:length(musc_names)
    errorbar(fig4_panelsA_I.stimAmp_uA, fig4_panelsA_I.(strcat(elec_configs(1), "_", musc_names(i), "_mean_SelIdx")), fig4_panelsA_I.(strcat(elec_configs(1), "_", musc_names(i), "_std_SelIdx")), "--", "Color", colors(i)); hold on
    errorbar(fig4_panelsA_I.stimAmp_uA, fig4_panelsA_I.(strcat(elec_configs(2), "_", musc_names(i), "_mean_SelIdx")), fig4_panelsA_I.(strcat(elec_configs(2), "_", musc_names(i), "_std_SelIdx")), "-", "Color", colors(i)); hold on
end
legend_str = repmat("", 2*length(musc_names), 1); legend_str(2:2:end) = musc_names;
legend(legend_str, 'Location', "southoutside", "Orientation", "horizontal", "NumColumns", 2);
xticks(500:500:1500)
xlim([450, 1550])
xlabel("EES amp (uA)")
ylabel("Selectivity Index (a.u.)")
title(tcl, "Monopolar (dashed) vs Narrow Bipolar (solid)")

figure()
% Plot mono vs wide rAUC and Selectivity Index
tcl = tiledlayout(1, 2);
nexttile
for i = 1:length(musc_names)
    errorbar(fig4_panelsA_I.stimAmp_uA, fig4_panelsA_I.(strcat(elec_configs(1), "_", musc_names(i), "_mean_rAUC")), fig4_panelsA_I.(strcat(elec_configs(1), "_", musc_names(i), "_std_rAUC")), "--", "Color", colors(i)); hold on
    errorbar(fig4_panelsA_I.stimAmp_uA, fig4_panelsA_I.(strcat(elec_configs(3), "_", musc_names(i), "_mean_rAUC")), fig4_panelsA_I.(strcat(elec_configs(3), "_", musc_names(i), "_std_rAUC")), "-", "Color", colors(i)); hold on
end
legend_str = repmat("", 2*length(musc_names), 1); legend_str(2:2:end) = musc_names;
legend(legend_str, 'Location', "southoutside", "Orientation", "horizontal", "NumColumns", 2);
xticks(500:500:1500)
xlim([450, 1550])
xlabel("EES amp (uA)")
ylabel("rAUC (a.u.)")

nexttile
for i = 1:length(musc_names)
    errorbar(fig4_panelsA_I.stimAmp_uA, fig4_panelsA_I.(strcat(elec_configs(1), "_", musc_names(i), "_mean_SelIdx")), fig4_panelsA_I.(strcat(elec_configs(1), "_", musc_names(i), "_std_SelIdx")), "--", "Color", colors(i)); hold on
    errorbar(fig4_panelsA_I.stimAmp_uA, fig4_panelsA_I.(strcat(elec_configs(3), "_", musc_names(i), "_mean_SelIdx")), fig4_panelsA_I.(strcat(elec_configs(3), "_", musc_names(i), "_std_SelIdx")), "-", "Color", colors(i)); hold on
end
legend_str = repmat("", 2*length(musc_names), 1); legend_str(2:2:end) = musc_names;
legend(legend_str, 'Location', "southoutside", "Orientation", "horizontal", "NumColumns", 2);
xticks(500:500:1500)
xlim([450, 1550])
xlabel("EES amp (uA)")
ylabel("Selectivity Index (a.u.)")
title(tcl, "Monopolar (dashed) vs Wide Bipolar (solid)")

figure()
% Plot narrow vs wide rAUC and Selectivity Index
tcl = tiledlayout(1, 2);
nexttile
for i = 1:length(musc_names)
    errorbar(fig4_panelsA_I.stimAmp_uA, fig4_panelsA_I.(strcat(elec_configs(3), "_", musc_names(i), "_mean_rAUC")), fig4_panelsA_I.(strcat(elec_configs(3), "_", musc_names(i), "_std_rAUC")), "--", "Color", colors(i)); hold on
    errorbar(fig4_panelsA_I.stimAmp_uA, fig4_panelsA_I.(strcat(elec_configs(2), "_", musc_names(i), "_mean_rAUC")), fig4_panelsA_I.(strcat(elec_configs(2), "_", musc_names(i), "_std_rAUC")), "-", "Color", colors(i)); hold on
end
legend_str = repmat("", 2*length(musc_names), 1); legend_str(2:2:end) = musc_names;
legend(legend_str, 'Location', "southoutside", "Orientation", "horizontal", "NumColumns", 2);
xticks(500:500:1500)
xlim([450, 1550])
xlabel("EES amp (uA)")
ylabel("rAUC (a.u.)")

nexttile
for i = 1:length(musc_names)
    errorbar(fig4_panelsA_I.stimAmp_uA, fig4_panelsA_I.(strcat(elec_configs(3), "_", musc_names(i), "_mean_SelIdx")), fig4_panelsA_I.(strcat(elec_configs(3), "_", musc_names(i), "_std_SelIdx")), "--", "Color", colors(i)); hold on
    errorbar(fig4_panelsA_I.stimAmp_uA, fig4_panelsA_I.(strcat(elec_configs(2), "_", musc_names(i), "_mean_SelIdx")), fig4_panelsA_I.(strcat(elec_configs(2), "_", musc_names(i), "_std_SelIdx")), "-", "Color", colors(i)); hold on
end
legend_str = repmat("", 2*length(musc_names), 1); legend_str(2:2:end) = musc_names;
legend(legend_str, 'Location', "southoutside", "Orientation", "horizontal", "NumColumns", 2);
xticks(500:500:1500)
xlim([450, 1550])
xlabel("EES amp (uA)")
ylabel("Selectivity Index (a.u.)")
title(tcl, "Wide Bipolar (dashed) vs Narrow Bipolar (solid)")
