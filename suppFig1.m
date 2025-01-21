%% suppFig1.m - The code to plot Supplementary Figure 1 of "An active electronic, high-density epidural paddle array for chronic spinal cord neuromodulation"
% Author: Samuel Parker
% Last Modified: 21-Jan-2025

clear all; close all; clc
load(strcat("data", filesep, "supFig1.mat"))

%% Tensile

figure("Renderer","painters")
tcl = tiledlayout(1, 3, "TileSpacing", "tight");
nexttile()
boxplot(tensile{:, ["singleMarkerTailPctElongation", "dualMarkerTailPctElongation"]} .* 100); hold on
scatter(ones(height(tensile), 1), tensile{:, "singleMarkerTailPctElongation"} .* 100, "MarkerEdgeColor", "none", "MarkerFaceColor", "#33bbee");
scatter(2.* ones(height(tensile), 1), tensile{:, "dualMarkerTailPctElongation"} .* 100, "MarkerEdgeColor", "none", "MarkerFaceColor", "#33bbee");
plot([1; 2] .* ones(1, height(tensile)), [tensile{:, "singleMarkerTailPctElongation"}'; tensile{:, "dualMarkerTailPctElongation"}'] .* 100, "Color", "#bbbbbb")
plot([0, 3], [5, 5], '--r')
title(tcl, sprintf("Elongation test | n = %d paddle arrays (%d tails)", height(tensile), height(tensile) * 2));
xticks([1, 2]);
xlabel("Lead-tail")
xticklabels(["Single marker", "Dual marker"])
ylabel("Percent elongation (%)")
ylim([0, 7.5])
nexttile()
boxplot(tensile{:, ["singleMarkerTailPctElongation", "dualMarkerTailPctElongation"]} .* 100); hold on
scatter(ones(height(tensile), 1), tensile{:, "singleMarkerTailPctElongation"} .* 100, "MarkerEdgeColor", "none", "MarkerFaceColor", "#33bbee");
scatter(2.* ones(height(tensile), 1), tensile{:, "dualMarkerTailPctElongation"} .* 100, "MarkerEdgeColor", "none", "MarkerFaceColor", "#33bbee");
plot([1; 2] .* ones(1, height(tensile)), [tensile{:, "singleMarkerTailPctElongation"}'; tensile{:, "dualMarkerTailPctElongation"}'] .* 100, "Color", "#bbbbbb")
plot([0, 3], [5, 5], '--r')
xticks([1, 2]);
xlabel("Lead-tail")
xticklabels(["Single marker", "Dual marker"])
ylabel("Percent elongation (%)")


%% Flex Testing

nexttile()
histogram(reshape(flex.connector{:, 3:14}, 1, [])); hold on
plot([6, 6], [0, 90], '--r')
title(sprintf("Flex test | n = %d lead tails", height(flex.connector)));
xticks([1:7]);
ylabel("Number of conductors")
xlabel("Resistance (Ohms)")
ylim([0, 90])
xlim([1, 7])
while(~all(get(gcf, "Position") == [117, 97, 1441, 420]))
    set(gcf, "Position", [117, 97, 1441, 420]);
    drawnow;
end