%% Import Data
AAPL = readtable('AAPL.csv');
AAPL.Spread = AAPL.AdjustedHigh - AAPL.AdjustedLow;
AAPL.LogSpread = log(AAPL.Spread);
AAPL.LogSpreadDiff = [NaN; diff(AAPL.LogSpread)];

% Getting MA5 and MA22
AAPL.WeeklyLogSpreadDiff = NaN(height(AAPL), 1);
AAPL.MonthlyLogSpreadDiff = NaN(height(AAPL), 1);
for i = 6:height(AAPL)
    AAPL.WeeklyLogSpreadDiff(i) = mean(AAPL.LogSpreadDiff(i-4:i));
    if i > 22
        AAPL.MonthlyLogSpreadDiff(i) = mean(AAPL.LogSpreadDiff(i-20:i));
    end
end

% Target Variable
AAPL.Target = [AAPL.LogSpreadDiff(2:end); NaN];

% Use non-NaN data
data = AAPL(23:end, ["LogSpread", "LogSpreadDiff", "WeeklyLogSpreadDiff", "MonthlyLogSpreadDiff", "Target"]);
%% Train-Test Split
trainProp = 0.8;
numObservations = height(data);
idxTrain = 1:floor(trainProp*numObservations);
idxTest = 1+floor(trainProp*numObservations):numObservations;
dataTrain = data(idxTrain, :);
dataTest = data(idxTest, :);

% Train Data
diffTrain = dataTrain(:, "LogSpreadDiff");
weeklyDiffTrain = dataTrain(:, "WeeklyLogSpreadDiff");
monthlyDiffTrain = dataTrain(:, "MonthlyLogSpreadDiff");
XTrain = [diffTrain, weeklyDiffTrain, monthlyDiffTrain];
YTrain = dataTrain.Target;

% Test Data
diffTest = dataTest(:, "LogSpreadDiff");
weeklyDiffTest = dataTest(:, "WeeklyLogSpreadDiff");
monthlyDiffTest = dataTest(:, "MonthlyLogSpreadDiff");
XTest = [diffTest, weeklyDiffTest, monthlyDiffTest];
YTest = dataTest.Target;
%% Modelling
model = fitlm(table2array(XTrain), YTrain);
predictedDiffLogSpread = predict(model, table2array(XTest));
predictedLogSpread = zeros(length(predictedDiffLogSpread), 1);
actualLogSpread = zeros(length(predictedDiffLogSpread), 1);

for i = 1:length(predictedDiffLogSpread)-1
    predictedLogSpread(i) = predictedDiffLogSpread(i) + dataTest.LogSpread(i);
    actualLogSpread(i) = YTest(i) + dataTest.LogSpread(i);
end

predictedSpread = exp(predictedLogSpread(1:end-1));
actualSpread = exp(actualLogSpread(1:end-1));
error = rmse(predictedSpread, actualSpread);
mape2 = mape(predictedSpread, actualSpread);