function fullName = getFeatureName(layer, seq)

pathName = 'F:\NN\MNIST_data\DBN\';
fileName = ['layer' num2str(layer) 'seq' num2str(seq) '.mat'];
fullName = [pathName fileName];

end