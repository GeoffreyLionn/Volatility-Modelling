%% Import Data
AAPL = readtable('AAPL.csv');
AAPL.Spread = AAPL.AdjustedHigh - AAPL.AdjustedLow;

% Log Transformation
data = log(AAPL.Spread);
dataDiff = diff(data);
%% Train-Test Split
seqLength = 5; % seqLength: number of previous days considered
trainProp = 0.8; % trainProp: proportion of training data
numObservations = numel(dataDiff) - seqLength;
idxTrain = 1:floor(trainProp*numObservations);
idxTest = 1+floor(trainProp*numObservations):seqLength+numObservations;
dataTrain = dataDiff(idxTrain);
dataTest = dataDiff(idxTest);

% Train Data
XTrain = zeros(height(dataTrain) - seqLength, seqLength);
YTrain = zeros(height(dataTrain) - seqLength, 1);
for i = 1:numel(dataTrain) - seqLength
    XTrain(i, :) = dataTrain(i:i+seqLength-1);
    YTrain(i) = dataTrain(i+seqLength);
end

% Test Data
XTest = zeros(numel(dataTest) - seqLength, seqLength);
YTest = zeros(numel(dataTest) - seqLength, 1);
for i = 1:numel(dataTest) - seqLength
    XTest(i, :) = dataTest(i:i+seqLength-1);
    YTest(i) = dataTest(i+seqLength);
end
%% Parameters
LearningRate = [0.001, 0.002, 0.005, 0.01];
drop = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
batch = [32, 64, 128, 256];
HiddenUnits = [64, 128, 256];
Solver = ["sgdm", "rmsprop", "adam"];
[LR, do, BS, HU, S] = ndgrid(LearningRate, drop, batch, HiddenUnits, Solver);
random = randperm(numel(LR), 100);
%% Tuning
variable_names = {'LearningRate', 'Dropout', 'BatchSize', 'HiddenUnits', 'Solver', 'Loss'};
variable_types = {'double', 'double', 'int32', 'int32', 'string', 'double'};

% Create empty table with variable names
T = table('Size', [0 numel(variable_names)], 'VariableNames', variable_names, 'VariableTypes', variable_types);

for iter = 1:100
    layers = [
        sequenceInputLayer(seqLength)
        lstmLayer(HU(random(iter)))
        dropoutLayer(do(random(iter)))
        fullyConnectedLayer(1)
    ];
    opts = trainingOptions(S(random(iter)), ...
        'ExecutionEnvironment','auto', ...
        'MaxEpochs',25, ...
        'Shuffle','every-epoch', ...
        'MiniBatchSize',BS(random(iter)), ...
        'InitialLearnRate', LR(random(iter)), ...
        'Verbose',false);
    [net, info] = trainnet(XTrain,YTrain,layers,"mse",opts);
    loss = info.TrainingHistory.Loss(end);
    fprintf('Iteration No: %d, Training Loss: %d\n', iter, loss);
    new_data = {LR(random(iter)), do(random(iter)), BS(random(iter)), HU(random(iter)), S(random(iter)), loss};
    T = [T; new_data];
end
%% Sort Table
sortedT = sortrows(T, 'Loss');
resultList = {};

% Get the results on the top 5 parameters
for j = 1:5
    layers_j = [
            sequenceInputLayer(seqLength)
            lstmLayer(sortedT.HiddenUnits(j))
            dropoutLayer(sortedT.Dropout(j))
            fullyConnectedLayer(1)
        ];
    
    % Model Parameters
    opts_j = trainingOptions(sortedT.Solver(j), ...
            'ExecutionEnvironment', 'auto', ...
            'MaxEpochs', 25, ...
            'Shuffle', 'every-epoch', ...
            'MiniBatchSize', sortedT.BatchSize(j), ...
            'InitialLearnRate', sortedT.LearningRate(j), ...
            'Verbose', false);
    
    % Final Model Training
    net_j = trainnet(XTrain, YTrain, layers_j, "mse", opts_j);
    resultList{j} = predict(net_j, XTest);
end

% Average of the 5 models
FinalPred = (cell2mat(resultList(1)) + cell2mat(resultList(2)) + cell2mat(resultList(3)) + cell2mat(resultList(4)) + ...
    cell2mat(resultList(5))) / 5;
%% Final Model Prediction

% Revert back the differencing
forecastedLogDiff = zeros(length(FinalPred), 1);
actualLogDiff = zeros(length(FinalPred), 1);
for i = 1:length(FinalPred)
    forecastedLogDiff(i) = data(floor(trainProp*numObservations) + seqLength + i) + FinalPred(i);
    actualLogDiff(i) = data(floor(trainProp*numObservations) + seqLength + i) + YTest(i);
end

prediction = exp(forecastedLogDiff);
actual = exp(actualLogDiff);
%% Error Evaluation
errorLSTMFinal = rmse(prediction, actual);
mapeLSTMFinal = mape(prediction, actual);