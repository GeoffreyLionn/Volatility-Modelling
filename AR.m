%% Import Data
AAPL = readtable('AAPL.csv');
AAPL.Spread = AAPL.AdjustedHigh - AAPL.AdjustedLow;
data = AAPL.Spread;
%% Train-Test Split (1: No Transformation and Differencing)
trainProp = 0.8;
numObservations = numel(data);
idxTrain = 1:floor(trainProp*numObservations);
idxTest = 1+floor(trainProp*numObservations):numObservations;
dataTrain = data(idxTrain);
dataTest = data(idxTest);

% Training
AR1 = arima(1,0,0);
[EstMdl,EstParamCov,logL,info] = estimate(AR1, dataTrain);
predicted = zeros(numel(dataTest, 1));

% Forecast first test based on the last value of train
predicted(1) = forecast(EstMdl, 1, dataTrain(end));

% Forecast test
for i=1:height(dataTest)-1
    predicted(i+1) = forecast(EstMdl, 1, dataTest(i));
end

% Get RMSE
predicted = predicted.';
RMSE_AR = rmse(predicted, dataTest);

% Plotting (Red: forecasted, Blue: actual)
plot(1:height(dataTest),predicted, 'r', 1:height(dataTest), dataTest, 'b');
%% Train-Test Split (2: Log Transformation without Differencing)
dataLog = log(AAPL.Spread);
numObservationsLog = numel(dataLog);
idxTrainLog = 1:floor(trainProp*numObservationsLog);
idxTestLog = 1+floor(trainProp*numObservationsLog):numObservationsLog;
dataTrainLog = dataLog(idxTrainLog);
dataTestLog = dataLog(idxTestLog);

% Training
AR1 = arima(1,0,0);
[EstMdlLog,EstParamCovLog,logLLog,infoLog] = estimate(AR1, dataTrainLog);
predictedLog = zeros(numel(dataTestLog, 1));

% Forecast first test based on the last value of train
predictedLog(1) = forecast(EstMdlLog, 1, dataTrainLog(end));

% Forecast Test
for i=1:height(dataTestLog)-1
    predictedLog(i+1) = forecast(EstMdlLog, 1, dataTestLog(i));
end

% Get RMSE
predictedLog = predictedLog.';
RMSE_AR_Log = rmse(exp(predictedLog), exp(dataTestLog));

% Plotting (Red: forecasted, Blue: actual)
plot(1:height(dataTestLog), predictedLog, 'r', 1:height(dataTestLog), dataTestLog, 'b');
%% Train-Test Split (3: Log Transformation with Differencing)
dataLog = log(AAPL.Spread);
dataLogDiff = diff(dataLog);
numObservationsLogDiff = numel(dataLogDiff);
idxTrainLogDiff = 1:floor(trainProp*numObservationsLogDiff);
idxTestLogDiff = 1+floor(trainProp*numObservationsLogDiff):numObservationsLogDiff;
dataTrainLogDiff = dataLogDiff(idxTrainLogDiff);
dataTestLogDiff = dataLogDiff(idxTestLogDiff);

% Training
AR1 = arima(1,0,0);
[EstMdlLogDiff,EstParamCovLogDiff,logLLogDiff,infoLogDiff] = estimate(AR1, dataTrainLogDiff);
predictedLogDiff = zeros(numel(dataTestLogDiff, 1));

% Forecast first test based on the last value of train
predictedLogDiff(1) = forecast(EstMdlLogDiff, 1, dataTrainLogDiff(end));

% Forecast Test
for i=1:height(dataTestLogDiff)-1
    predictedLogDiff(i+1) = forecast(EstMdlLogDiff, 1, dataTestLogDiff(i));
end

predictedLogDiff = predictedLogDiff.';

% Revert back the differencing
forecastedLogDiff = zeros(length(predictedLogDiff),1);
actualLogDiff = zeros(length(predictedLogDiff),1);
for i = 1:length(predictedLogDiff)
    forecastedLogDiff(i) = dataLog(floor(trainProp*numObservationsLogDiff) + i) + predictedLogDiff(i);
    actualLogDiff(i) = dataLog(floor(trainProp*numObservationsLogDiff) + i) + dataTestLogDiff(i);
end

% Get RMSE
RMSE_AR_Log_Diff = rmse(exp(forecastedLogDiff), exp(actualLogDiff));

% Plotting (Red: forecasted, Blue: actual)
plot(1:height(forecastedLogDiff), exp(forecastedLogDiff), 'r', 1:height(forecastedLogDiff), exp(actualLogDiff), 'b');