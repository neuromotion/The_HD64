%% fig1.m - The code to plot figure 1 of "An active electronic, high-density epidural paddle array for chronic spinal cord neuromodulation"
% Author: Samuel Parker
% Last Modified: 25-Sep-2024

clear all; close all; clc
%% Panel G

load("fig1_panelG.mat")
MIL_spec_leak_rate_air = 5e-9;
MIL_spec_leak_rate_He = MIL_spec_leak_rate_air * 2.7; % http://www.leakdetection-technology.com/science/the-flow-of-gases-in-leaks/conversion-of-helium-leak-rate-to-air-leak-rate.html

figure()
leak_rate = str2double(hermeticity.LHe_Leakage_Rate_atm_cc_per_sec);
leak_rate(isnan(leak_rate)) = 1.1e-10;
histogram(leak_rate, 0:2.2e-10:((2.2e-10)*13)); hold on
plot([MIL_spec_leak_rate_He, MIL_spec_leak_rate_He], [0, height(hermeticity)], '--r')
plot([limit_of_detection, limit_of_detection], [0, height(hermeticity)], '-.r')
title("Hermetic Package Leak rate")
subtitle(sprintf("n = %d", height(hermeticity)))
ylabel("Number of active packages")
xlabel("Leak Rate (L_{He} atm-cm^3/s)")
legend("HD64", sprintf("MIL-STD-883K (%0.3g)", MIL_spec_leak_rate_He), "Limit of Detection")
set(gcf, "Position", [100, 100, 560, 420])