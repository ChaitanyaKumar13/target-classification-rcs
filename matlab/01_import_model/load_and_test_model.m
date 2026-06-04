clc;
clear;

disp('STEP 3: Load CNN model from Python');

projectRoot = 'C:\Users\chait\OneDrive\Desktop\target-classification-rcs';

if count(py.sys.path, projectRoot) == 0
    insert(py.sys.path, int32(0), projectRoot);
end

disp('Project root added to Python path');

tf = py.importlib.import_module('tensorflow');

disp('TensorFlow import OK');
disp('TensorFlow version:');
disp(char(tf.version));

disp('STEP 3 COMPLETE');