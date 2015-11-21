% testBP

load('weight.mat');
load('label_test_.mat');

pathName = 'F:\NN\MNIST\test';
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

format compact
[result',output'];
errRate = errNum/testNum;