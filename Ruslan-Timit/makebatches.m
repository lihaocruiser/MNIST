% need external parameters:
% -- batchSize
% -- phone: mfcc feature, input of NN
% -- label: speaker id, output of NN

mfccFileName = 'dr1.mat';
batchSize = 1000;
load(mfccFileName);

speakerNum = max(max(max(label)));
dimen = size(phone, 1);
numBatch = size(phone, 2) / batchSize;
numBatch = floor(numBatch);
numBatchTest = ceil(numBatch/10);
numBatchTest = 0;
numBatchTrain = numBatch - numBatchTest;
labelOut = zeros(speakerNum, size(phone,2));
indexSum = 1;
for ii = 1:speakerNum
    num = length(find(label == 1));
    % curLabel = [zeros(ii-1, num); ones(1, num); zeros(speakerNum-ii, num)];   % set target value
    curLabel = [0.05*ones(ii-1, num); 0.95*ones(1, num); 0.05*ones(speakerNum-ii, num)];   % set target value
    labelOut(:, indexSum:indexSum+num-1) = curLabel;
    indexSum = indexSum + num;
end

batchdata = zeros(batchSize, dimen, numBatchTrain);
batchtargets = zeros(batchSize, speakerNum, numBatchTrain);
testbatchdata = zeros(batchSize, dimen, numBatchTest);
testbatchtargets = zeros(batchSize, speakerNum, numBatchTest);

randomseq = randperm(size(phone, 2));
phone = phone(:, randomseq);
labelOut = labelOut(:, randomseq);

for ii = 1:numBatchTrain
    s0 = batchSize * (ii-1) + 1;
    s1 = batchSize * ii;
    batchdata(:, :, ii) = phone(:, s0:s1)';
    batchtargets(: , :, ii) = labelOut(:, s0:s1)';
end

for ii = 1:numBatchTest
    s0 = batchSize * (ii-1) + 1;
    s1 = batchSize * ii;
    testbatchdata(:, :, ii) = phone(:, s0:s1)';
    testbatchtargets(:, :, ii) = labelOut(:, s0:s1)';
end
