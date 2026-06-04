function predict_batch_carrada()
clc; clear;

disp("STEP 7: Batch prediction on Carrada RD frames → save preds.csv")

% --- paths ---
projectRoot = "C:\Users\chait\OneDrive\Desktop\target-classification-rcs";
outputsDir  = fullfile(projectRoot, "outputs");
carradaRoot = fullfile(projectRoot, "data", "raw", "Carrada");

addpath(genpath(fullfile(projectRoot,"matlab")));

% --- load model if not already ---
if evalin('base','exist("model","var")') ~= 1
    disp("Model not found in workspace, loading...");
    load_trained_model;
end
model = evalin('base',"model");

% --- python imports ---
np = py.importlib.import_module("numpy");

% --- choose how many frames to predict ---
MAX_FRAMES = 300;   % change to 1000 later if you want
classNames = ["car","pedestrian","cyclist"];

% --- collect RD npy files (across sequences) ---
seqDirs = dir(carradaRoot);
seqDirs = seqDirs([seqDirs.isdir]);
seqNames = string({seqDirs.name});
seqNames = seqNames(~ismember(seqNames,[".",".."]));

allFiles = strings(0);
allSeqs  = strings(0);

for s = 1:length(seqNames)
    rdDir = fullfile(carradaRoot, seqNames(s), "range_doppler_numpy");
    if ~isfolder(rdDir), continue; end
    f = dir(fullfile(rdDir,"*.npy"));
    if isempty(f), continue; end

    for k = 1:length(f)
        allFiles(end+1) = fullfile(rdDir, f(k).name); %#ok<AGROW>
        allSeqs(end+1)  = seqNames(s);                %#ok<AGROW>
    end
end

if isempty(allFiles)
    error("No .npy RD files found. Check carradaRoot path.");
end

% --- sample subset for speed ---
rng(42);
idx = randperm(length(allFiles), min(MAX_FRAMES, length(allFiles)));
allFiles = allFiles(idx);
allSeqs  = allSeqs(idx);

disp("Total frames selected: " + string(length(allFiles)))

% --- output table holders ---
seqCol   = strings(length(allFiles),1);
fileCol  = strings(length(allFiles),1);
predCol  = strings(length(allFiles),1);
p0 = zeros(length(allFiles),1);
p1 = zeros(length(allFiles),1);
p2 = zeros(length(allFiles),1);

% --- run predictions ---
for i = 1:length(allFiles)
    fp = allFiles(i);
    [~, fname, ext] = fileparts(fp);

    % load numpy array
    arr = np.load(fp);
    x = double(arr);

    % normalize per-sample
    x = (x - mean(x(:))) / (std(x(:)) + 1e-6);

    % reshape to (1,256,64,1)
    x = reshape(x, [1, size(x,1), size(x,2), 1]);

    % to python numpy
    x_np = np.array(x);

    % predict
    pred = model.predict(x_np);
    pred_mat = double(pred);
    pred_mat = pred_mat(:)';

    [~, clsIdx] = max(pred_mat);
    clsName = classNames(clsIdx);

    % store
    seqCol(i)  = allSeqs(i);
    fileCol(i) = fname + ext;
    predCol(i) = clsName;
    p0(i) = pred_mat(1);
    p1(i) = pred_mat(2);
    p2(i) = pred_mat(3);

    if mod(i,50)==0
        disp("Done: " + string(i) + "/" + string(length(allFiles)))
    end
end

T = table(seqCol, fileCol, predCol, p0, p1, p2, ...
    'VariableNames', {'sequence','frame','predicted','p_car','p_pedestrian','p_cyclist'});

outFile = fullfile(projectRoot,"matlab","02_run_inference_on_carrada","preds.csv");
writetable(T, outFile);

disp("✅ Saved predictions CSV: " + outFile)
disp("STEP 7 COMPLETE ✅")
end
