function generate_sim_dataset()
clc; clear;

disp("STEP 10: Generating synthetic defence dataset (RD maps) -> .npy + labels.csv");

%% ====== Paths ======
projectRoot = "C:\Users\chait\OneDrive\Desktop\target-classification-rcs";
outRoot = fullfile(projectRoot, "matlab", "04_train_with_sim_data", "sim_data_npy");

if ~exist(outRoot, 'dir')
    mkdir(outRoot);
end

%% ====== Python (NumPy) access for saving .npy ======
% Assumes you've already set pyenv to rcsenv python earlier.
np = py.importlib.import_module("numpy");

%% ====== Dataset settings ======
Nr = 256;  % range bins
Nd = 64;   % doppler bins

classes = ["drone","truck","tank"];
nPerClass = 800;                 % change to 500/1000 as you want
rng(42);                          % reproducibility

% Output folders
for c = 1:numel(classes)
    classDir = fullfile(outRoot, classes(c));
    if ~exist(classDir,'dir')
        mkdir(classDir);
    end
end

%% ====== Label CSV (MATLAB table) ======
seq = strings(nPerClass*numel(classes),1);  % will store filename
lab = strings(nPerClass*numel(classes),1);  % will store label
k = 1;

%% ====== Generate loop ======
for c = 1:numel(classes)
    cls = classes(c);
    classDir = fullfile(outRoot, cls);

    disp("Generating: " + cls + " (" + nPerClass + " samples)");

    for i = 1:nPerClass

        % ---- Randomize target parameters per sample ----
        % Range center / Doppler center
        r0 = randi([20, 235]);         % keep away from edges
        d0 = randi([5, 60]);

        % Noise / clutter level
        noiseSigma = 0.8 + rand()*0.7; % ~0.8 to 1.5

        % Signature params per class
        switch cls
            case "drone"
                amp = 8 + rand()*6;      % moderate
                sr = 2 + rand()*2;       % narrow in range
                sd = 10 + rand()*12;     % wide doppler (micro-doppler band)
            case "truck"
                amp = 12 + rand()*8;     % stronger than drone
                sr = 3 + rand()*3;       % medium range spread
                sd = 3 + rand()*3;       % compact doppler
            case "tank"
                amp = 18 + rand()*10;    % strongest
                sr = 2 + rand()*2;       % compact blob
                sd = 2 + rand()*2;       % low doppler
        end

        % ---- Build RD map ----
        RD = noiseSigma * randn(Nr, Nd);

        % Add smooth clutter bands (optional mild realism)
        if rand() < 0.35
            bandR = randi([10, Nr-10]);
            RD(bandR:bandR+2, :) = RD(bandR:bandR+2, :) + (0.5 + rand());
        end

        % Add target "blob" / signature
        RD = RD + make_blob(Nr, Nd, r0, d0, sr, sd, amp);

        % Drone: add extra horizontal micro-doppler smear sometimes
        if cls == "drone" && rand() < 0.7
            smearWidth = randi([1,3]);
            RD(max(1,r0-smearWidth):min(Nr,r0+smearWidth), :) = ...
                RD(max(1,r0-smearWidth):min(Nr,r0+smearWidth), :) + (0.6 + rand());
        end

        % ---- Normalize (same idea as your inference preprocessing) ----
        RD = (RD - mean(RD(:))) / (std(RD(:)) + 1e-6);

        % ---- Save as .npy ----
        fname = sprintf("%s_%04d.npy", cls, i);
        fpath = fullfile(classDir, fname);

        % Convert to python numpy and save
        % Ensure float32 for consistency with CNN input
        arr = single(RD);
        np.save(fpath, py.numpy.array(arr));

        % ---- record label row ----
        seq(k) = string(fname);
        lab(k) = cls;
        k = k + 1;

        if mod(i, 100) == 0
            disp("  " + cls + ": " + i + "/" + nPerClass);
        end
    end
end

%% ====== Write labels.csv ======
T = table(seq, lab, 'VariableNames', {'file','label'});
csvPath = fullfile(outRoot, "labels.csv");
writetable(T, csvPath);

disp("✅ STEP 10 COMPLETE: Saved dataset to:");
disp(outRoot);
disp("✅ labels.csv saved to:");
disp(csvPath);

end

%% ====== Helper: 2D Gaussian blob ======
function B = make_blob(Nr, Nd, r0, d0, sr, sd, amp)
% returns Nr x Nd blob centered at (r0,d0)

[r, d] = ndgrid(1:Nr, 1:Nd);
B = amp * exp(-(((r - r0).^2)/(2*sr^2) + ((d - d0).^2)/(2*sd^2)));

end
