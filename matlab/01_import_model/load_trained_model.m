clc;
clear;

disp("STEP 4: Load trained CNN model (best_model.keras) from outputs/");

% --- 1) Set your project root (edit if needed) ---
projectRoot = "C:\Users\chait\OneDrive\Desktop\target-classification-rcs";
outputsDir  = fullfile(projectRoot, "outputs");

if ~isfolder(outputsDir)
    error("outputs folder not found: " + outputsDir);
end

% --- 2) Ensure Python can see your project root ---
if count(py.sys.path, char(projectRoot)) == 0
    insert(py.sys.path, int32(0), char(projectRoot));
end

% --- 3) Import TensorFlow ---
tf = py.importlib.import_module("tensorflow");
disp("TensorFlow imported OK.");

% --- 4) Find latest cnn_run_* folder ---
d = dir(outputsDir);
isDir = [d.isdir];
d = d(isDir);

% remove . and .. (string-safe)
names = string({d.name});
d = d(~ismember(names, ["." ".."]));

% keep only cnn_run_*
keep = false(size(d));
for i = 1:numel(d)
    keep(i) = startsWith(string(d(i).name), "cnn_run_");
end
d = d(keep);

if isempty(d)
    error("No cnn_run_* folders found inside outputs.");
end


% sort by date (newest first)
[~, idx] = sort([d.datenum], "descend");
d = d(idx);

latestRun = fullfile(outputsDir, d(1).name);
disp("Latest run folder: " + string(latestRun));

% --- 5) Choose model file (prefer best_model.keras) ---
bestPath  = fullfile(latestRun, "best_model.keras");
finalPath = fullfile(latestRun, "final_model.keras");

if isfile(bestPath)
    modelPath = bestPath;
    disp("Using model: best_model.keras");
elseif isfile(finalPath)
    modelPath = finalPath;
    disp("best_model.keras not found. Using final_model.keras instead.");
else
    error("No best_model.keras or final_model.keras found in: " + string(latestRun));
end

disp("Model path: " + string(modelPath));

% --- 6) Load model in Python/TensorFlow ---
% Import tensorflow.keras properly (avoid KerasLazyLoader issues)
keras = py.importlib.import_module("tensorflow.keras");
modelsMod = py.importlib.import_module("tensorflow.keras.models");

% Load model
model = modelsMod.load_model(modelPath);
disp(" Model loaded successfully (via tensorflow.keras.models.load_model).")

% --- 7) Quick sanity check: print input/output shape ---
inpShape = model.input_shape;
outShape = model.output_shape;

disp("Model input_shape (Python):");
disp(inpShape);

disp("Model output_shape (Python):");
disp(outShape);

% --- 8) Save variables to MATLAB workspace for next scripts ---
assignin("base", "tf", tf);
assignin("base", "model", model);
assignin("base", "latestRun", latestRun);
assignin("base", "modelPath", modelPath);

disp("STEP 4 COMPLETE: model is now in MATLAB workspace as variable 'model'.");
