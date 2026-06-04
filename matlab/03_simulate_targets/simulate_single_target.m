clc; clear;

disp("STEP 8: Simulating ONE synthetic radar target (RD map)");

% RD map size (same as Carrada)
Nr = 256;   % range bins
Nd = 64;    % doppler bins

RD = zeros(Nr, Nd);

% --- target parameters ---
target_range_bin   = 120;   % mid-range
target_doppler_bin = 40;    % moving target
rcs_strength       = 15;    % signal strength

% Inject target
RD(target_range_bin, target_doppler_bin) = rcs_strength;

% Add clutter + noise
RD = RD + 0.5 * randn(size(RD));

% Normalize
RD = (RD - mean(RD(:))) / (std(RD(:)) + 1e-6);

% Display
figure;
imagesc(RD);
colormap jet;
colorbar;
xlabel("Doppler bins");
ylabel("Range bins");
title("Synthetic Radar Range–Doppler Map (Single Target)");

disp("STEP 8 COMPLETE: synthetic RD target generated ✅");
