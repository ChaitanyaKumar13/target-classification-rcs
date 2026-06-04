function test_inference_matlab()
clc; clearvars -except model; close all;

fprintf("STEP 5: Quick inference test from MATLAB\n");

% If model isn't in workspace, load it
if ~exist("model","var")
    fprintf("Model not found in workspace, loading...\n");
    load_trained_model;  % calls your Step 4 script
end

% --- Create a dummy radar frame (256x64x1) ---
x = rand(256,64,1,'single');    % dummy input
x = (x - mean(x(:))) / (std(x(:)) + 1e-6); % normalize like training

% Add batch dimension -> (1,256,64,1)
x_batch = reshape(x, [1,256,64,1]);

% Convert MATLAB array -> numpy
np = py.importlib.import_module("numpy");
x_np = np.array(x_batch);

% Predict
pred = model.predict(x_np);

% Convert output to MATLAB
pred_mat = double(pred);
pred_mat = pred_mat(:)';

fprintf("Predicted probabilities: [%.4f  %.4f  %.4f]\n", pred_mat(1), pred_mat(2), pred_mat(3));

% Class mapping (same order as your Python config)
class_names = ["car","pedestrian","cyclist"];
[~, idx] = max(pred_mat);
fprintf("Predicted class: %s\n", class_names(idx));

fprintf("STEP 5 COMPLETE ✅\n");
end
