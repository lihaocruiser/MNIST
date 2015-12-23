% Version 1.000
%
% Code provided by LiHao
%
% Substitution for backpropclassify.m
% Using mini-batch gradient decent.

maxepoch = 300;

load mnistvhclassify
load mnisthpclassify
load mnisthp2classify

makebatches;
[numcases, numdims, numbatches]=size(batchdata);
N=numcases; 
speakerNum = size(batchtargets, 2);

algorithm = 'SGD';  % CGD or SGD
holdNum = 5;        % First update top-level weights holding other weights fixed. 
etaHold = 2;
etaStart = [2 1 0.5 0.1] * 10;
etaFinal = [2 1 0.5 0.1] / 2;
weightcost = 0.0002;
combineFactor = 1;
trainBatchNum = floor(numbatches / combineFactor);
trainBatchSize = numcases;

fprintf(1,'Training discriminative model on TIMIT by minimizing cross entropy error. \n');
fprintf(1,'%s, %d batches of %d cases each. \n', algorithm, trainBatchNum, trainBatchSize);

%%%% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w1=[vishid; hidrecbiases];
w2=[hidpen; penrecbiases];
w3=[hidpen2; penrecbiases2];
w4 = 0.1*randn(size(w3,2)+1, speakerNum);
 

%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l1=size(w1,1)-1;
l2=size(w2,1)-1;
l3=size(w3,1)-1;
l4=size(w4,1)-1;
l5=speakerNum; 

test_err=zeros(1,maxepoch);
train_err=zeros(1,maxepoch);
test_crerr=zeros(1,maxepoch);
train_crerr=zeros(1,maxepoch);

% pureinc1 = zeros(trainBatchSize, size(w1,1), size(w1,2));
% pureinc2 = zeros(trainBatchSize, size(w2,1), size(w2,2));
% pureinc3 = zeros(trainBatchSize, size(w3,1), size(w3,2));
% pureinc4 = zeros(trainBatchSize, size(w4,1), size(w4,2));

for epoch = 1:maxepoch

    tic;
    %%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    err=0; 
    err_cr=0;
    counter=0;
    [numcases, numdims, numbatches]=size(batchdata);
    N=numcases;
    parfor batch = 1:numbatches
        data = batchdata(:,:,batch);
        target = batchtargets(:,:,batch);
        data = [data ones(N,1)];
        w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
        w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
        w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
        targetout = exp(w3probs*w4);
        targetout = targetout./repmat(sum(targetout,2),1,speakerNum);

        [I, J] = max(targetout,[],2);
        [I1, J1] = max(target,[],2);
        counter = counter + length(find(J==J1));
        err_cr = err_cr - sum(sum( target(:,1:end).*log(targetout))) ;
    end
    train_err(epoch)=(numcases*numbatches-counter);
    train_crerr(epoch)=err_cr/numbatches;

    %%%%%%%%%%%%%% END OF COMPUTING TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%% COMPUTE TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    err_cr=0;
    counter=0;
    [testnumcases, testnumdims, testnumbatches]=size(testbatchdata);
    N=testnumcases;
    parfor batch = 1:testnumbatches
        data = testbatchdata(:,:,batch);
        target = testbatchtargets(:,:,batch);
        data = [data ones(N,1)];
        w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
        w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
        w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
        targetout = exp(w3probs*w4);
        targetout = targetout./repmat(sum(targetout,2),1,speakerNum);

        [I, J]=max(targetout,[],2);
        [I1, J1]=max(target,[],2);
        counter=counter+length(find(J==J1));
        err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
    end
    test_err(epoch)=(testnumcases*testnumbatches-counter);
    test_crerr(epoch)=err_cr/testnumbatches;
    fprintf(1,'Before epoch %2d. # Train Error:%5d (%d). # Test Error:%4d (%d). ',...
            epoch,train_err(epoch),numcases*numbatches,test_err(epoch),testnumcases*testnumbatches);
    %%%%%%%%%%%%%% END OF COMPUTING TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for batch = 1:trainBatchNum
        % fprintf(1,'epoch %d batch %d\r',epoch,batch);

        %%%%%%%%%%% COMBINE MINIBATCHES INTO LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data=[];
        targets=[]; 
        for kk=1:combineFactor
            data=[data, batchdata(:,:,(batch-1)*combineFactor+kk)]; 
            targets=[targets, batchtargets(:,:,(batch-1)*combineFactor+kk)];
        end 

        %%%%%%%%%%%%%%% PERFORM STOCHASTIC GRADIENT DECENT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if (strcmp(algorithm,'SGD'))
            if epoch<=holdNum
                N = size(data,1);
                XX = [data ones(N,1)];
                w1probs = 1./(1 + exp(-XX*w1));      w1probs = [w1probs  ones(N,1)];
                w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
                w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs ones(N,1)];

                targetout = 1./(1 + exp(-w3probs*w4));
                % targetout = exp(w3probs*w_class);
                % targetout = targetout./repmat(sum(targetout,2),1,speakerNum);

                delta4 = targetout - targets;
                w4 = w4 - etaHold / trainBatchSize * w3probs' * delta4 - etaHold * weightcost * w4;
            else
                if (epoch<=maxepoch*0.8)
                    eta = etaStart;
                else
                    eta = etaFinal;
                end
                N = size(data,1);
                XX = [data ones(N,1)];
                w1probs = 1./(1 + exp(-XX*w1));      w1probs = [w1probs, ones(N,1)];
                w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs, ones(N,1)];
                w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs, ones(N,1)];

                targetout = 1./(1 + exp(-w3probs*w4));
                % targetout = exp(w3probs*w_class);
                % targetout = targetout./repmat(sum(targetout,2),1,speakerNum);
                

                delta4 = targetout - targets;
                delta3 = delta4 * w4' .* romsigmoidy(w3probs); delta3 = delta3(:,1:end-1);
                delta2 = delta3 * w3' .* romsigmoidy(w2probs); delta2 = delta2(:,1:end-1);
                delta1 = delta2 * w2' .* romsigmoidy(w1probs); delta1 = delta1(:,1:end-1);

                w4 = w4 - eta(4) * (w3probs' * delta4 / trainBatchSize - weightcost * w4);
                w3 = w3 - eta(3) * (w2probs' * delta3 / trainBatchSize - weightcost * w3);
                w2 = w2 - eta(2) * (w1probs' * delta2 / trainBatchSize - weightcost * w2);
                w1 = w1 - eta(1) * (XX'      * delta1 / trainBatchSize - weightcost * w1);
            end
        end
        %%%%%%%%%%%%%%% END OF STOCHASTIC GRADIENT DECENT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         if (strcmp(algorithm,'CGD'))
%             max_iter=3;
%             if epoch<holdNum  % First update top-level weights holding other weights fixed. 
%                 N = size(data,1);
%                 XX = [data ones(N,1)];
%                 w1probs = 1./(1 + exp(-XX*w1)); w1probs = [w1probs  ones(N,1)];
%                 w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
%                 w3probs = 1./(1 + exp(-w2probs*w3)); %w3probs = [w3probs  ones(N,1)];
% 
%                 VV = [w_class(:)']';
%                 Dim = [l4; l5];
%                 [X, fX] = minimize(VV,'CG_CLASSIFY_INIT',max_iter,Dim,w3probs,targets);
%                 w_class = reshape(X,l4+1,l5);
%             else
%                 VV = [w1(:)' w2(:)' w3(:)' w_class(:)']';
%                 Dim = [l1; l2; l3; l4; l5];
%                 [X, fX] = minimize(VV,'CG_CLASSIFY',max_iter,Dim,data,targets);
% 
%                 w1 = reshape(X(1:(l1+1)*l2),l1+1,l2);
%                 xxx = (l1+1)*l2;
%                 w2 = reshape(X(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
%                 xxx = xxx+(l2+1)*l3;
%                 w3 = reshape(X(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
%                 xxx = xxx+(l3+1)*l4;
%                 w_class = reshape(X(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
%             end
%         end
        %%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
	toc;

	save mnistclassify_weights w1 w2 w3 w4
 	save mnistclassify_error test_err test_crerr train_err train_crerr;

end

fprintf('%s\n', datestr(now));