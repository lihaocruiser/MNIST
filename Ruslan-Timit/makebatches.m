load('feature1.mat');
batchSize = 100;

mfcc = mfcc(2:13, :);
mfcc = mfcc - min(min(mfcc));
mfcc = mfcc / max(max(mfcc));

speakerNum = max(max(max(label)));
dimen = size(mfcc, 1);
numBatch = size(mfcc, 2) / batchSize;
numBatch = floor(numBatch);
numBatchTest = ceil(numBatch/10);
numBatchTrain = numBatch - numBatchTest;
labelOut = zeros(speakerNum, size(mfcc,2));
indexSum = 1;
for ii = 1:speakerNum
    num = length(find(label == 1));
    curLabel = [zeros(ii-1, num); ones(1, num); zeros(speakerNum-ii, num)];
    labelOut(:, indexSum:indexSum+num-1) = curLabel;
    indexSum = indexSum + num;
end

batchdata = zeros(batchSize, dimen, numBatchTrain);
batchtargets = zeros(batchSize, speakerNum, numBatchTrain);
testbatchdata = zeros(batchSize, dimen, numBatchTest);
testbatchtargets = zeros(batchSize, speakerNum, numBatchTest);

randomseq = randperm(size(mfcc, 2));
mfcc = mfcc(:, randomseq);
labelOut = labelOut(:, randomseq);

for ii = 1:numBatchTrain
    s0 = batchSize * (ii-1) + 1;
    s1 = batchSize * ii;
    batchdata(:, :, ii) = mfcc(:, s0:s1)';
    batchtargets(: , :, ii) = labelOut(:, s0:s1)';
end

for ii = 1:numBatchTest
    s0 = batchSize * (ii-1) + 1;
    s1 = batchSize * ii;
    testbatchdata(:, :, ii) = mfcc(:, s0:s1)';
    testbatchtargets(:, :, ii) = labelOut(:, s0:s1)';
end
