tic;

% pathName = 'F:\NN\MNIST_data\DBN';
load('F:\NN\MNIST_data\label\label_train.mat'); % d
NUM = [28*28, 512, 10];
numTrain = length(d);
eta = 0.1;

%% init
w1 = randn(NUM(2), NUM(1));
b1 = randn(NUM(2), 1);

w2 = randn(NUM(3), NUM(2));
b2 = randn(NUM(3), 1);

% w3 = randn(NUM(3), NUM(2));
% b3 = randn(NUM(3), 1);

%% pre-train input layer

for ii = 1:numTrain
    feature = getTrainPic(ii);
    p = logsig(w1 * feature + b1);
    hidden = p > rand(size(p));
    w1 = w1 - eta * hidden * feature';
    b1 = b1 - eta * hidden;
end

for ii = 1:numTrain
    feature = getTrainPic(ii);
    p = logsig(w1 * feature + b1);
    feature = p > rand(size(p));
    save(getFeatureName(1, ii), 'feature');
end

%% pre-train hidden layer

for ii = 1:numTrain
    load(getFeatureName(1, numTrain));
    p = logsig(w2 * feature + b2);
    hidden = p > rand(size(p));
    w2 = w2 - eta * hidden * feature';
    b2 = b2 - eta * hidden;
end

for ii = 1:numTrain
    fileName = getFeatureName(1, ii);
    load(fileName);
    p = logsig(w2 * feature + b2);
    feature = p > rand(size(p));
    save(getFeatureName(2, ii), 'feature');
end


%% BP

for ii = 1:numTrain

    pic = getTrainPic(ii);
    
    % output 10 * 1
    res = zeros(NUM(3), 1);
    res(d(ii)+1) = 1;
    
    % calculate output
    a1 = w1 * pic + b1;
    z1 = logsig(a1);
    a2 = w2 * z1 + b2;
    z2 = a2;

    % back propagation
    delta2 = z2 - res;
    delta1 = w2' * delta2 .* romsigmoid(a1);

    % update
    w2 = w2 - eta * delta2 * z1';
    b2 = b2 - eta * delta2;
    w1 = w1 - eta * delta1 * pic';
    b1 = b1 - eta * delta1;
end


%% Test

pathName = 'F:\NN\MNIST_data\test';
numTest = 10000;
result = d( 1 : numTest );
errNum = 0;

output = zeros(1, numTest);
for j = 1:numTest
    fileName = [pathName '\test' num2str(j) '.bmp'];
    pic = imread(fileName, 'bmp');
    pic = reshape(pic, NUM(1), 1);
    pic = (1 - double(pic) / 255) / 100;
    a1 = w1 * pic + b1;
    z1 = logsig(a1);
    a2 = w2 * z1 + b2;
    z2 = a2;
    [~, index] = max(z2);
    output(j) = index - 1;
    if(output(j)~=result(j))
        errNum = errNum + 1;
    end
end

[result',output'];
errRate = errNum/numTest;

fprintf('errRate = %1.4f. ', errRate);

toc;