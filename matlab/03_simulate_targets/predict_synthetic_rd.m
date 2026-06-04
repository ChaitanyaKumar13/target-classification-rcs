% matlab/03_simulate_targets/predict_synthetic_rd.m
clc; clear;

disp("STEP 9: Predict CNN on synthetic RD (MATLAB -> Python model.predict)");

% -----------------------------
% 0) Ensure paths are available
% -----------------------------
projectRoot = "C:\Users\chait\OneDrive\Desktop\target-classification-rcs";

% Add all matlab subfolders to MATLAB path (only if not already)
addpath(genpath(fullfile(projectRoot, "matlab")));

% Make sure Python can import your project root (optional but safe)
if count(py.sys.path, projectRoot) == 0
    insert(py.sys.path, int32(0), projectRoot);
end

% -----------------------------
% 1) Make sure model exists
% -----------------------------
if ~evalin('base','exist("model","var")')
    disp("Model not found in base workspace, loading...");
    load_trained_model; % loads variable 'model' in base workspace
end

model = evalin('base','model');

% -----------------------------
% 2) Choose which synthetic class to generate
% -----------------------------
% Change this to "drone" / "truck" / "tank"
targetClass = "tank";

% RD shape expected by CNN: 256 x 64
Nr = 256; Nd = 64;

% -----------------------------
% 3) Generate synthetic RD map
%    (simple physics-inspired signatures)
% -----------------------------
RD = zeros(Nr, Nd);

% background noise
RD = RD + 0.25 * randn(Nr, Nd);

% target parameters (pick per class)
switch lower(targetClass)
    case "drone"
        % small RCS, long doppler smear (moving rotors -> spread)
        r0 = 150; d0 = 30;
        amp = 10;  sr = 2;  sd = 10;

    case "truck"
        % medium RCS, compact blob
        r0 = 90; d0 = 48;
        amp = 14; sr = 5;  sd = 3;

    case "tank"
        % high RCS, strong return, low doppler (heavier/slow)
        r0 = 40; d0 = 5;
        amp = 22; sr = 3;  sd = 2;

    otherwise
        error("Unknown class. Use drone/truck/tank.");
end

% 2D Gaussian blob
[rr, dd] = ndgrid(1:Nr, 1:Nd);
blob = amp * exp(-((rr - r0).^2)/(2*sr^2) - ((dd - d0).^2)/(2*sd^2));
RD = RD + blob;

% -----------------------------
% 4) Normalize EXACTLY like your CNN pipeline
%    (per-sample z-score across H,W)
% -----------------------------
mu = mean(RD(:));
sigma = std(RD(:)) + 1e-6;
RDn = (RD - mu) ./ sigma;

% -----------------------------
% 5) Convert to CNN input shape: (1,256,64,1)
% -----------------------------
x = single(reshape(RDn, [1, Nr, Nd, 1]));

% Convert MATLAB array -> NumPy array -> TensorFlow tensor
np = py.importlib.import_module("numpy");
tf = py.importlib.import_module("tensorflow");

x_np = np.array(x);
x_tf = tf.convert_to_tensor(x_np);

% -----------------------------
% 6) Predict
% -----------------------------
pred = model.predict(x_tf);     % returns py.numpy.ndarray
pred_mat = double(pred);        % convert to MATLAB double

probs = pred_mat(1, :);

class_names = ["car", "pedestrian", "cyclist"];
[~, idx] = max(probs);

fprintf("\nSynthetic target generated as: %s\n", targetClass);
fprintf("Predicted probabilities: [%.4f  %.4f  %.4f]\n", probs(1), probs(2), probs(3));
fprintf("Predicted class (CNN told): %s\n", class_names(idx));

% -----------------------------
% 7) Visualize synthetic RD
% -----------------------------
figure;
imagesc(RDn);
axis xy;
colorbar;
title("Synthetic RD (normalized) - " + targetClass);
xlabel("Doppler bins");
ylabel("Range bins");

disp("STEP 9 COMPLETE ✅");
