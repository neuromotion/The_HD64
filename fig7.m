%% fig7.m - The code to plot figure 7 of "An active electronic bidirectional interface for high resolution interrogation of the spinal cord"
% Author: Samuel Parker
% Last Modified: 21-May-2024

clear all; close all; clc
%% Panel B

load("fig7_panelB.mat")
figure()
boxplot(fig7_panelB.L1_error, fig7_panelB.condition);
xlabel("Model Name")
ylabel("L1 error on forward model test dataset")

%% Panel C

load("fig7_panelC.mat")
figure()
boxplot(fig7_panelC.L1_error, fig7_panelC.condition);
xlabel("Model Name")
ylabel("L1 error on forward model test dataset")

%% Panel D

load("fig7_panelD.mat")
figure()
boxplotGroup(fig7_panelD.L1_errs, 'primaryLabels', fig7_panelD.model_coverage_labels, 'SecondaryLabels', fig7_panelD.muscle_labels);
ylabel("L1 error on forward model test dataset")