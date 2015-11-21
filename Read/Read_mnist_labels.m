% 用于读取MNIST数据集中t10k-images.idx3-ubyte文件并将其转换成bmp格式图片输出。
% 用法：运行程序，会弹出选择测试图片数据文件t10k-labels.idx1-ubyte路径的对话框和
% 选择保存测试图片路径的对话框，选择路径后程序自动运行完毕，期间进度条会显示处理进度。
% 图片以TestImage_00001.bmp～TestImage_10000.bmp的格式保存在指定路径，10000个文件占用空间39M。。
% 整个程序运行过程需几分钟时间。
% Written By DXY@HUST IPRAI
% 2009-2-22
clear all;
clc;
%读取训练图片数据文件
[FileName,PathName] = uigetfile('*.*','选择测试图片数据文件');
TrainFile = fullfile(PathName,FileName);
fid = fopen(TrainFile,'r'); %fopen（）是最核心的函数，导入文件，‘r’代表读入
a = fread(fid,8,'uint8'); %这里需要说明的是，包的前十六位是说明信息，从上面提到的那个网页可以看到具体那一位代表什么意义。所以a变量提取出这些信息，并记录下来，方便后面的建立矩阵等动作。
MagicNum = ((a(1)*256+a(2))*256+a(3))*256+a(4)
ImageNum = ((a(5)*256+a(6))*256+a(7))*256+a(8)
%从上面提到的网页可以理解这四句
d = zeros(1,ImageNum);
for i = 1:ImageNum
    d(i) = fread(fid,1);
end
save('test_label.mat', 'd');
% savedirectory = uigetdir('','选择测试图片路径：');
% savepath = fullfile(savedirectory,['label']);
% fod = fopen(savepath, 'w');
% for i=1:ImageNum
%     b = fread(fid,1);   %fread（）也是核心的函数之一，b记录下了一副图的数据串。注意这里还是个串，是看不出任何端倪的。
%     fwrite(fod, b);
% end
fclose(fid);
% fclose(fod);