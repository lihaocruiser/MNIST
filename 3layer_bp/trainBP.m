tic;

pathName = 'F:\NN\MNIST_data\train';
load('F:\NN\MNIST_data\label\label_train.mat');
picSize = 28 * 28;
trainNum = length(d);
N1 = 800;

N = [picSize N1 10];

lambda = 0.1;
numIter = 15;
weightNum = N(1) * N(2) + N(2) * N(3);
%decayFactor = 1 - eta * lambda / weightNum;
decayFactor = 1;
eta = 0.1;

trainErrors = zeros(1, numIter);
testErrors = zeros(1, numIter);
etas = zeros(1, numIter);

% init weight1
w1 = randn(N(2), N(1));
b1 = randn(N(2), 1);

% init weight2
w2 = randn(N(3), N(2));
b2 = randn(N(3), 1);

for iter = 1:numIter

%     if (iter <= 2 )
%         eta = 0.2;
%     elseif (iter <= 4)
%         eta = 0.05;
%     elseif (iter <= 6)
%         eta = 0.02;
%     end
    
for ii = 1:trainNum
    
    % input 764 * 1
    pic = getTrainPic(ii);
    
    % output 10 * 1
    load('F:\NN\MNIST_data\label\label_train.mat');
    res = zeros(N(3), 1);
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
    w2 = w2 * decayFactor - eta * delta2 * z1';
    b2 = b2 - eta * delta2;
    w1 = w1 * decayFactor  - eta * delta1 * pic';
    b1 = b1 - eta * delta1;
end

save weight w1 b1 w2 b2

etas(iter) = eta;
eta = eta * 0.9;

% CalTrainError;
% trainErrors(iter) = trainError;

CalTestError;
testErrors(iter) = testError;

end

figure; hold on; grid on;
plot(1:numIter, trainErrors, '-ro');
plot(1:numIter, testErrors, '-bo');
