load('F:\NN\MNIST_data\label\label_test.mat');

testNum = 10000;
result = d( 1 : testNum );
errNum = 0;

output = zeros(1, testNum);
for jj = 1:testNum
    pic = getTestPic(jj);
    a1 = w1 * pic + b1;
    z1 = logsig(a1);
    a2 = w2 * z1 + b2;
    z2 = a2;
    [~, index] = max(z2);
    output(jj) = index - 1;
    if(output(jj)~=result(jj))
        errNum = errNum + 1;
    end
end

testError = errNum/testNum;

fprintf('TestError = %1.4f. %s\n', testError, datestr(now));
