tic;
err = [];
for N1 = 12:15
    trainBP;
    testBP;
    err = [err, errRate];
    fprintf('N1= %2.0f, errRate= %1.4f,%s. ', N1, errRate, datestr(now));
    toc;
end

err