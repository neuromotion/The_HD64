%% fig5.m - The code to plot figure 5 of "An active electronic, high-density epidural paddle array for chronic spinal cord neuromodulation"
% Author: Samuel Parker
% Last Modified: 27-Sep-2024

clear all; close all; clc
%% Panel B

load(strcat("data", filesep, "fig5_panelB.mat"))
figure()
tiledlayout(2, 1);
nexttile();
boxplot(fig5_panelB.top.L1_error, fig5_panelB.top.condition);
xlabel("Model Name")
ylabel("L1 error on forward model test dataset")
title("L1 error by model")
nexttile();
boxplot(fig5_panelB.bottom.L1_error, fig5_panelB.bottom.condition);
xlabel("Model Name")
ylabel("L1 error on forward model test dataset")
title("L1 error by electrode inclusion")

%% Panel C

load(strcat("data", filesep, "fig5_panelC.mat"))
coverage = ["25", "50", "100"];
af_max_likelihood = 0;
elec_max_likelihood = 0;
for i = 1:length(coverage)
    af_max_likelihood = max([af_max_likelihood, max(fig5_panelC.(strcat("cov", coverage(i))).amp_freq.likelihood, [], 'all')]);
    elec_max_likelihood = max([elec_max_likelihood, max(fig5_panelC.(strcat("cov", coverage(i))).elec.likelihood, [], 'all')]);
end

figure()
tcl = tiledlayout(3, 2);
for i = 1:length(coverage)
    nexttile
    surf(fig5_panelC.(strcat("cov", coverage(i))).amp_freq.freq, fig5_panelC.(strcat("cov", coverage(i))).amp_freq.amp, fig5_panelC.(strcat("cov", coverage(i))).amp_freq.likelihood);
    view([90, -90]);
    shading interp
    title(strcat(coverage(i), "% Amp-Freq"))
    xlim([min(fig5_panelC.(strcat("cov", coverage(i))).amp_freq.freq, [], 'all'), max(fig5_panelC.(strcat("cov", coverage(i))).amp_freq.freq, [], 'all')])
    ylim([min(fig5_panelC.(strcat("cov", coverage(i))).amp_freq.amp, [], 'all'), max(fig5_panelC.(strcat("cov", coverage(i))).amp_freq.amp, [], 'all')])
    clim([0, af_max_likelihood]);
    xlabel("Frequency (Hz)")
    ylabel("Amplitude (uA)")
    nexttile
    surf(fig5_panelC.(strcat("cov", coverage(i))).elec.X, fig5_panelC.(strcat("cov", coverage(i))).elec.Y, fig5_panelC.(strcat("cov", coverage(i))).elec.likelihood);
    view([90, -90]);
    shading interp
    title(strcat(coverage(i), "% Electrode"))
    xlim([min(fig5_panelC.(strcat("cov", coverage(i))).elec.X, [], 'all'), max(fig5_panelC.(strcat("cov", coverage(i))).elec.X, [], 'all')])
    ylim([min(fig5_panelC.(strcat("cov", coverage(i))).elec.Y, [], 'all'), max(fig5_panelC.(strcat("cov", coverage(i))).elec.Y, [], 'all')])
    clim([0, elec_max_likelihood]);
end
colormap abyss

%% Panel E

load(strcat("data", filesep, "fig5_panelE.mat"))
figure()
boxplot(fig5_panelE.L1_Error, fig5_panelE.Coverage)
xlabel("Model")
ylabel("L1 Error (a.u.)")
