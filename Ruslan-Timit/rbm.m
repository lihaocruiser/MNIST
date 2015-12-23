% Version 1.000 
%
% Code provided by Geoff Hinton and Ruslan Salakhutdinov 
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.
% The program assumes that the following variables are set externally:

% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning

% results are stored in: vishid, visbiases, hidbiases

maxepoch  = 100;
epsilonw  = 0.01;   % Learning rate for weights 
epsilonvb = 0.01;   % Learning rate for biases of visible units 
epsilonhb = 0.01;   % Learning rate for biases of hidden units 

weightcost  = 0.0002;   
initialmomentum  = 0.5;
finalmomentum    = 0.9;

[numcases numdims numbatches]=size(batchdata);

if restart ==1,
    restart=0;
    epoch=1;

    % Initializing symmetric weights and biases. 
    vishid     = 0.1*randn(numdims, numhid);
    hidbiases  = zeros(1,numhid);
    visbiases  = zeros(1,numdims);

    poshidprobs = zeros(numcases,numhid);   % h
    neghidprobs = zeros(numcases,numhid);   % h'
    posprods    = zeros(numdims,numhid);    % positive outer-product
    negprods    = zeros(numdims,numhid);    % negative outer-product
    vishidinc  = zeros(numdims,numhid);     % delta w
    hidbiasinc = zeros(1,numhid);           % delta b
    visbiasinc = zeros(1,numdims);          % delta a
    batchposhidprobs=zeros(numcases,numhid,numbatches);
end

errsumvec = zeros(1,maxepoch);

for epoch = epoch:maxepoch,
    % fprintf(1,'epoch %d\r',epoch); 
    errsum=0;

    for batch = 1:numbatches,
        % fprintf(1,'epoch %d batch %d\r',epoch,batch); 

        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data = batchdata(:,:,batch);    % actual data
        poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));    
        batchposhidprobs(:,:,batch) = poshidprobs;
        posprods  = data' * poshidprobs;
        poshidact = sum(poshidprobs);
        posvisact = sum(data);
        
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        poshidstates = poshidprobs > rand(numcases,numhid);
        negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));   % 1-step reconstruction data
        neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));    
        negprods  = negdata'*neghidprobs;
        neghidact = sum(neghidprobs);
        negvisact = sum(negdata); 
        
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        err= sum(sum( (data-negdata).^2 ));
        errsum = err + errsum;

        if epoch>5,
            momentum = finalmomentum;
        else
        	momentum = initialmomentum;
        end;

        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        vishidinc  = momentum * vishidinc  + epsilonw  * ((posprods-negprods) / numcases - weightcost*vishid );
        visbiasinc = momentum * visbiasinc + epsilonvb * (posvisact-negvisact) / numcases;
        hidbiasinc = momentum * hidbiasinc + epsilonhb * (poshidact-neghidact) / numcases;

        vishid = vishid + vishidinc;
        visbiases = visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;

        %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    end
    fprintf(1, 'epoch %d error %6.1f.\n', epoch, errsum);
    errsumvec(epoch) = errsum;
end

% figure;plot(errsumvec);
fprintf('%s \n', datestr(now));
