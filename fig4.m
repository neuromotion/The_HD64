%% fig4.m - The code to plot figure 4 of "An active electronic, high-density epidural paddle array for chronic spinal cord neuromodulation"
% Author: Samuel Parker
% Last Modified: 27-Sep-2024

clear all; close all; clc
%% Panel C

load(strcat("data", filesep, "fig4_panelC.mat"))
colors = hex2rgb(["#bff4ff", "#80e9ff", "00d3ff"]);

figure()
tcl = tiledlayout(4, 1);
trial_stats = grpstats(fig4_panelC.data, ["Distance_mm", "EES_Amp_mA"], ["mean", "std"]);
unique_bipoles = unique(trial_stats.Distance_mm);
for i = length(unique_bipoles):-1:1
    this_bipole_data = trial_stats(trial_stats.Distance_mm == unique_bipoles(i), :);
    nexttile
    unique_amps = unique(this_bipole_data.EES_Amp_mA);
    for ampIdx = 1:length(unique_amps)    
        upper_bound = this_bipole_data{this_bipole_data.EES_Amp_mA == unique_amps(ampIdx), 6:2:end} + this_bipole_data{this_bipole_data.EES_Amp_mA == unique_amps(ampIdx), 7:2:end};
        lower_bound = this_bipole_data{this_bipole_data.EES_Amp_mA == unique_amps(ampIdx), 6:2:end} - this_bipole_data{this_bipole_data.EES_Amp_mA == unique_amps(ampIdx), 7:2:end};
        time2 = [fig4_panelC.time_us, fliplr(fig4_panelC.time_us)];
        shadedArea = [upper_bound, fliplr(lower_bound)];
        fill(time2, shadedArea, colors(ampIdx, :), "FaceAlpha", 0.5); hold on
        plot(fig4_panelC.time_us, this_bipole_data{this_bipole_data.EES_Amp_mA == unique_amps(ampIdx), 6:2:end}, "Color", "#000000");
    end
    title(sprintf("Bipole Center Distance: %0.2fmm", unique_bipoles(i)));
    xlim([0, 500])
    ylim([-0.225, 0.225])
    xticks(0:100:500)
    xlabel("Time (us)")
    ylabel("LFP (mV)")

end

%% Panel D

load(strcat("data", filesep, "fig4_panelD.mat"))

unique_amps = unique(fig4_panelD.eesAmp_mA);
figure()
tcl = tiledlayout(3, 1);
for ampIdx = 1:length(unique_amps)
    this_amp = unique_amps(ampIdx);

    nexttile
    gscatter(fig4_panelD.onsetTime_us(fig4_panelD.eesAmp_mA == this_amp), fig4_panelD.distance_mm(fig4_panelD.eesAmp_mA == this_amp), fig4_panelD.distance_mm(fig4_panelD.eesAmp_mA == this_amp));
    xlabel("Peak Onset Time [us]")
    ylabel("Distance from Stimulation Bipole [mm]");
    title(sprintf("%d mA", this_amp));
end
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