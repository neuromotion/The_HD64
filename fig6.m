%% fig6.m - The code to plot figure 6 of "An active electronic bidirectional interface for high resolution interrogation of the spinal cord"
% Author: Samuel Parker
% Last Modified: 21-May-2024

clear all; close all; clc

%% Panel B

load("fig6_panelB.mat")

figure()
tcl = tiledlayout(3, 1, "TileSpacing", "none");
mono_mask = startsWith(string(fig6_panelB.Properties.VariableNames), "mono");
hd64_mask = startsWith(string(fig6_panelB.Properties.VariableNames), "HD64");
mdt565_mask = startsWith(string(fig6_panelB.Properties.VariableNames), "mdt565");

nexttile
plot(fig6_panelB.time_s * 1000, fig6_panelB{:, mono_mask})
xlim([-25, 100]);
xticks(0:25:100)
yticks([])
ylabel({"Monopolar"; "Norm. LFP (a.u.)"})
xlabel("Time (ms)")
nexttile
plot(fig6_panelB.time_s * 1000, fig6_panelB{:, hd64_mask})
xlim([-25, 100]);
xticks(0:25:100)
yticks([])
ylabel({"Bipolar HD64"; "Norm. LFP (a.u.)"})
xlabel("Time (ms)")
nexttile
plot(fig6_panelB.time_s * 1000, fig6_panelB{:, mdt565_mask})
xlim([-25, 100]);
xticks(0:25:100)
yticks([])
ylabel({"Bipolar 565"; "Norm. LFP (a.u.)"})
xlabel("Time (ms)")

%% Panel C

load("fig6_panelC.mat")
figure()
boxplot(fig6_panelC.uniqueness, fig6_panelC.electrode);
