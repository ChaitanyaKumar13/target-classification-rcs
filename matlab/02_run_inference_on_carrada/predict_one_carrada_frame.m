function predict_one_carrada_frame()
clc; clearvars -except model; close all;

fprintf("STEP 6: Predict on ONE real Carrada RD frame (.npy)\n");

% Load model if not already in workspace
if ~exist("model","var")
    fprintf("Model not found in workspace, loading...\n");
    load_trained_model;
end

% --- Paths (EDIT IF YOUR DATA FOLDER NAME DIFFERS) ---
projectRoot = fileparts(fileparts(pwd)); % assumes you're inside matlab/...
carradaRoot = fullfile(projectRoot, "data", "raw", "Carrada");  % your logs show this folder exists

% Find first sequence folder
seqList = dir(carradaRoot);
seqList = seqList([seqList.isdir]);
seqNames = string({seqList.name});
seqNames = seqNames(~ismember(seqNames, [".",".."]));
if isempty(seqNames)
    error("No sequence folders found in: %s", carradaRoot);
end
seq = seqNames(1);

% Range-Doppler numpy folder
rdFolder = fullfile(carradaRoot, seq, "range_doppler_numpy");
if ~exist(rdFolder, "dir")
    error("range_doppler_numpy not found: %s", rdFolder);
end

% Pick the first .npy file
npyFiles = dir(fullfile(rdFolder, "*.npy"));
if isempty(npyFiles)
    error("No .npy files found in: %s", rdFolder);
end
npyPath = fullfile(npyFiles(1).folder, npyFiles(1).name);

fprintf("Using sequence: %s\n", seq);
fprintf("Using RD file: %s\n", npyFiles(1).name);
fprintf("Full path: %s\n", npyPath);

% --- Load .npy using Python numpy ---
np = py.importlib.import_module("numpy");
arr = np.load(npyPath);

% Convert numpy array -> MATLAB double
x = double(arr);

% Carrada RD frames sometimes come as (1,256,64) or (256,64)
% Convert to (256,64,1)
if ndims(x) == 3 && size(x,1) == 1
    x = squeeze(x(1,:,:));
end

if ~isequal(size(x), [256 64])
    fprintf("WARNING: unexpected size %dx%d. Trying to reshape/crop...\n", size(x,1), size(x,2));
end

% Force expected shape
x = single(x);
x = x(1:256, 1:64);        % safe crop if bigger (only works if >= expected)
x = reshape(x, [256,64,1]);

% Normalize (same as training: per-sample z-score)
x = (x - mean(x(:))) / (std(x(:)) + 1e-6);

% Add batch dimension -> (1,256,64,1)
x_batch = reshape(x, [1,256,64,1]);

% Convert to numpy and predict
x_np = np.array(x_batch);
pred = model.predict(x_np);

pred_mat = double(pred);
pred_mat = pred_mat(:)';

fprintf("Predicted probabilities: [%.4f  %.4f  %.4f]\n", pred_mat(1), pred_mat(2), pred_mat(3));

class_names = ["car","pedestrian","cyclist"];
[~, idx] = max(pred_mat);
fprintf("Predicted class: %s\n", class_names(idx));

% Optional: show RD frame as image
figure;
imagesc(squeeze(x)); axis image; colormap jet; colorbar;
title("Carrada RD frame (normalized)");

fprintf("STEP 6 COMPLETE ✅\n");
end
