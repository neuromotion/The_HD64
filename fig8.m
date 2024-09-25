%% fig8.m - The code to plot figure 8 of "An active electronic bidirectional interface for high resolution interrogation of the spinal cord"
% Author: Samuel Parker
% Last Modified: 21-May-2024

close all; clc
%% Panel A

load("fig8_panelA.mat")
coverage = ["25", "50", "100"];
af_max_likelihood = 0;
elec_max_likelihood = 0;
for i = 1:length(coverage)
    af_max_likelihood = max([af_max_likelihood, max(fig8_panelA.(strcat("cov", coverage(i))).amp_freq.likelihood, [], 'all')]);
    elec_max_likelihood = max([elec_max_likelihood, max(fig8_panelA.(strcat("cov", coverage(i))).elec.likelihood, [], 'all')]);
end

figure()
tcl = tiledlayout(3, 2);
for i = 1:length(coverage)
    nexttile
    surf(fig8_panelA.(strcat("cov", coverage(i))).elec.X, fig8_panelA.(strcat("cov", coverage(i))).elec.Y, fig8_panelA.(strcat("cov", coverage(i))).elec.likelihood);
    view([90, -90]);
    shading interp
    title(strcat(coverage(i), "% Electrode"))
    xlim([min(fig8_panelA.(strcat("cov", coverage(i))).elec.X, [], 'all'), max(fig8_panelA.(strcat("cov", coverage(i))).elec.X, [], 'all')])
    ylim([min(fig8_panelA.(strcat("cov", coverage(i))).elec.Y, [], 'all'), max(fig8_panelA.(strcat("cov", coverage(i))).elec.Y, [], 'all')])
    clim([0, elec_max_likelihood]);
    nexttile
    surf(fig8_panelA.(strcat("cov", coverage(i))).amp_freq.freq, fig8_panelA.(strcat("cov", coverage(i))).amp_freq.amp, fig8_panelA.(strcat("cov", coverage(i))).amp_freq.likelihood);
    view([90, -90]);
    shading interp
    title(strcat(coverage(i), "% Amp-Freq"))
    xlim([min(fig8_panelA.(strcat("cov", coverage(i))).amp_freq.freq, [], 'all'), max(fig8_panelA.(strcat("cov", coverage(i))).amp_freq.freq, [], 'all')])
    ylim([min(fig8_panelA.(strcat("cov", coverage(i))).amp_freq.amp, [], 'all'), max(fig8_panelA.(strcat("cov", coverage(i))).amp_freq.amp, [], 'all')])
    clim([0, af_max_likelihood]);
    xlabel("Frequency (Hz)")
    ylabel("Amplitude (uA)")
end
colormap abyss

%% Panel D

load("fig8_panelD.mat")
figure()
boxplot(fig8_panelD.L1_Error, fig8_panelD.Coverage)
xlabel("Model")
ylabel("L1 Error (a.u.)")

