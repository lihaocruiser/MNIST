% UP-DOWN ALGORITHM
% Geoffrey Hinton 2005
%
% the data and all biases are row vectors.
% the generative model is: lab <--> top <--> pen --> hid --> vis
% the number of units in layer foo is numfoo
% weight matrices have names fromlayer tolayer
% "rec" is for recognition biases and "gen" is for generative
% biases.
% for simplicity, the same learning rate, r, is used everywhere.

% perform a bottom-up pass to get wake/positive phase
% probabilities and sample states
wakeHidProbs = logsig(data*vishid + hidrecbiases);
wakeHidStates = wakeHidProbs > rand(1, numhid);
wakePenProbs = logsig(wakeHidStates*hidpen + penrecbiases);
wakePenStates = wakePenProbs > rand(1, numpen);
wakeTopprobs = logsig(wakePenStates*pentop + targets*labtop + topbiases);
wakeTopStates = wakeTopprobs > rand(1, numtop);

% positive phase statistics for contrastive devergence
PosLabTopStatistics = targets'* wakeTopStates;
PosPenTopStatistics = wakePenStates' * wakeTopStates;

% perform numCDiters gibbs sampling iterations using the top level
% undirected associative memory
negTopStates = wakeTopStates;   % to initialize loop
for iter=1:numCDiters
negpenprobs = logsig(negTopStates*pentop' + pengenbiases);
negpenstates = negpenprobs > rand(1, numpen);
neglabprobs = softmax(negTopStates*labtop' + labgenbiases);
negtopprobs = logsig(negpenstates*pentop+neglabprobs*labtop + topbiases);
negTopStates = negtopprobs > rand(1, numtop);
end;

% negative phase statistics for contrastive divergence
negpentopstatistics = negpenstates'*negTopStates;
neglabtopstatistics = neglabprobs'*negTopStates;

% starting from the end of the gibbs sampling run, perfrom a
% top-down generative pass to get sleep/negative phase
% probabilities and sample states
sleeppenstates = negpenstates;
sleephidprobs = logsig(sleeppenstates*penhid + hidgenbiases);
sleephidstates = sleephidprobs > rand(1, numhid);
sleepvisprobs = logsig(sleephidstates*hidvis + visgenbiases);

% predictions
psleeppenstates = logsig(sleephidstates*hidpen + penrecbiases);
psleephidstates = logsig(sleepvisprobs*vishid + hidrecbiases);
pvisprobs = logsig(wakeHidStates*hidvis + visgenbiases);
phidprobs = logsig(wakePenStates*penhid + hidgenbiases);

% UPDATES TO GENERATIVE PARAMETERS
hidvis = hidvis + r*poshidstates'*(data-pvisprobs);

visgenbiases = visgenbiases + r*(data - pvisprobs);
penhid = penhid + r*wakePenStates'*(wakeHidStates-phidprobs);
hidgenbiases = hidgenbiases + r*(wakeHidStates - phidprobs);

% UPDATES TO TOP LEVEL ASSOCIATIVE MEMORY PARAMETERS
labtop = labtop + r*(PosLabTopStatistics-neglabtopstatistics);
labgenbiases = labgenbiases + r*(targets - neglabprobs);
pentop = pentop + r*(PosPenTopStatistics - negpentopstatistics);
pengenbiases = pengenbiases + r*(wakePenStates - negpenstates);
topbiases = topbiases + r*(wakeTopStates - negTopStates);

%UPDATES TO RECOGNITION/INFERENCE APPROXIMATION PARAMETERS
hidpen = hidpen + r*(sleephidstates'*(sleeppenstatespsleeppenstates));
penrecbiases = penrecbiases + r*(sleeppenstates-psleeppenstates);
vishid = vishid + r*(sleepvisprobs'*(sleephidstatespsleephidstates));
hidrecbiases = hidrecbiases + r*(sleephidstates-psleephidstates);