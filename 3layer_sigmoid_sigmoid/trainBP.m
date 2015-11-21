%% classify MNIST

%% train BP

tic;

pathName = 'F:\NN\MNIST_data\train';
load('F:\NN\MNIST_data\label\label_train.mat');
picSize = 28 * 28;
trainNum = length(d);
N1 = 15;
eta = 0.5;

N = [picSize N1 10];

lambda = 0.1;
weightNum = N(1) * N(2) + N(2) * N(3);
decayFactor = 1 - eta * lambda / weightNum;
decayFactor = 1;

% init weight1
w1 = randn(N(2), N(1));
b1 = randn(N(2), 1);

% init weight2
w2 = randn(N(3), N(2));
b2 = randn(N(3) ,1);

for monte = 1:4

for i = 1:trainNum
    
    % input 764 * 1
    fileName = [pathName '\train' num2str(i) '.bmp'];
    pic = imread(fileName, 'bmp');
    pic = reshape(pic, N(1), 1);
    pic = (1 - double(pic) / 255) / 100;
    
    % output 10 * 1
    res = zeros(N(3), 1);
    res(d(i)+1) = 1;
    
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

eta = eta/2;
end

save('weight','w1','b1','w2','b2','trainNum','N');

toc;

%% test BP

load('F:\NN\MNIST_data\label\label_test.mat');

pathName = 'F:\NN\MNIST_data\test';
testNum = 10000;
result = d( 1 : testNum );
errNum = 0;

output = zeros(1, testNum);
for j = 1:testNum
    fileName = [pathName '\test' num2str(j) '.bmp'];
    pic = imread(fileName, 'bmp');
    pic = reshape(pic, N(1), 1);
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
errRate = errNum/testNum;

fprintf('errRate = %1.4f. ', errRate);

toc;