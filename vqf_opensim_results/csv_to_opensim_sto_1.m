%% Convert VQF CSV Orientations to OpenSim .sto Format

clear; close all; clc;

%% Configuration
outputFile = 'imu_orientations.sto';
opensimVersion = '4.4';
AUTO_DETECT = true;

csvFiles_manual = {
    'pelvis_imu.csv';
    'femur_r_imu.csv';
    'femur_l_imu.csv';
    'tibia_r_imu.csv';
    'tibia_l_imu.csv';
    'calcn_r_imu.csv';
    'calcn_l_imu.csv';
};

%% Detect or load files
if AUTO_DETECT
    fileList = dir('*_imu.csv');
    if isempty(fileList)
        error('No *_imu.csv files found in current directory');
    end
    [~, sortIdx] = sort({fileList.name});
    fileList = fileList(sortIdx);
    csvFiles = {fileList.name}';
else
    existingFiles = {};
    for i = 1:length(csvFiles_manual)
        if exist(csvFiles_manual{i}, 'file')
            existingFiles{end+1} = csvFiles_manual{i};
        end
    end
    csvFiles = existingFiles';
end

if isempty(csvFiles)
    error('No CSV files to process');
end

%% Load data
imuData = cell(length(csvFiles), 1);
imuNames = cell(length(csvFiles), 1);

for i = 1:length(csvFiles)
    data = readtable(csvFiles{i});
    requiredCols = {'time', 'qw', 'qx', 'qy', 'qz'};
    if ~all(ismember(requiredCols, data.Properties.VariableNames))
        error('File %s missing required columns', csvFiles{i});
    end
    imuData{i} = data;
    [~, name, ~] = fileparts(csvFiles{i});
    imuNames{i} = name;
end

%% Synchronize data
allSampleCounts = cellfun(@(x) height(x), imuData);
minSamples = min(allSampleCounts);

for i = 1:length(imuData)
    if height(imuData{i}) > minSamples
        imuData{i} = imuData{i}(1:minSamples, :);
    end
end

refTime = imuData{1}.time;
numSamples = minSamples;

%% Calculate data rate
dt = mean(diff(refTime));
dataRate = round(1/dt);

%% Write .sto file
fid = fopen(outputFile, 'w');

fprintf(fid, 'DataRate=%.6f\n', dataRate);
fprintf(fid, 'DataType=Quaternion\n');
fprintf(fid, 'version=3\n');
fprintf(fid, 'OpenSimVersion=%s\n', opensimVersion);
fprintf(fid, 'endheader\n');

fprintf(fid, 'time');
for i = 1:length(imuNames)
    fprintf(fid, '\t%s', imuNames{i});
end
fprintf(fid, '\n');

for frameIdx = 1:numSamples
    fprintf(fid, '%.6f', refTime(frameIdx));
    for imuIdx = 1:length(imuData)
        data = imuData{imuIdx};
        fprintf(fid, '\t%.16f,%.16f,%.16f,%.16f', ...
            data.qw(frameIdx), data.qx(frameIdx), ...
            data.qy(frameIdx), data.qz(frameIdx));
    end
    fprintf(fid, '\n');
end

fclose(fid);

fprintf('Success: Created %s with %d IMUs and %d frames\n', ...
    outputFile, length(imuNames), numSamples);