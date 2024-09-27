%% fig2.m - The code to plot figure 2 of "An active electronic, high-density epidural paddle array for chronic spinal cord neuromodulation"
% Author: Samuel Parker
% Last Modified: 27-Sep-2024

clear all; close all; clc
%% Panel D

load("fig2_panelD.mat")
day_labels = string(day_nums);
[patches, L, means, meds] = violin(impedances, 'mc', [], 'medc', []);
set(L, "visible", 'off');
for i = length(patches):-1:1
    patches(i).XData = (patches(i).XData - i) * (max(day_nums) / length(day_nums)/2) + day_nums(i);
    patches(i).FaceColor = [51, 188, 238] ./ 255;
    patches(i).FaceAlpha = 1;
end
xlim([0, 300])
xticks(0:30:300)
xticklabels(string(0:30:300))
hold on
for i = 1:length(means)
    plot([day_nums(i)-5, day_nums(i)+5], repmat(means(i), 1, 2), '-k')
    plot([day_nums(i)-5, day_nums(i)+5], repmat(meds(i), 1, 2), '-r')
end
set(gcf, "Position", [2349         441        1037         420]);
xlabel("Days since implant")
ylabel("Impedance (k\Omega{})")