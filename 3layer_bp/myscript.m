tic;
err = [];
for eta = 0.1
    trainBP;
    err = [err, errRate];
    fprintf('errRate= %1.4f,%s. ', errRate, datestr(now));
    toc;
end