function pic = getTestPic(ii)

pathName = 'F:\NN\MNIST_data\test\';
fileName = ['test' num2str(ii) '.bmp'];
fullName = [pathName fileName];

pic = imread(fullName, 'bmp');
pic = reshape(pic, 784, 1);
pic = (1 - double(pic) / 255) / 100;

end