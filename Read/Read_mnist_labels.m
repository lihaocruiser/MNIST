% ���ڶ�ȡMNIST���ݼ���t10k-images.idx3-ubyte�ļ�������ת����bmp��ʽͼƬ�����
% �÷������г��򣬻ᵯ��ѡ�����ͼƬ�����ļ�t10k-labels.idx1-ubyte·���ĶԻ����
% ѡ�񱣴����ͼƬ·���ĶԻ���ѡ��·��������Զ�������ϣ��ڼ����������ʾ������ȡ�
% ͼƬ��TestImage_00001.bmp��TestImage_10000.bmp�ĸ�ʽ������ָ��·����10000���ļ�ռ�ÿռ�39M����
% �����������й����輸����ʱ�䡣
% Written By DXY@HUST IPRAI
% 2009-2-22
clear all;
clc;
%��ȡѵ��ͼƬ�����ļ�
[FileName,PathName] = uigetfile('*.*','ѡ�����ͼƬ�����ļ�');
TrainFile = fullfile(PathName,FileName);
fid = fopen(TrainFile,'r'); %fopen����������ĵĺ����������ļ�����r���������
a = fread(fid,8,'uint8'); %������Ҫ˵�����ǣ�����ǰʮ��λ��˵����Ϣ���������ᵽ���Ǹ���ҳ���Կ���������һλ����ʲô���塣����a������ȡ����Щ��Ϣ������¼�������������Ľ�������ȶ�����
MagicNum = ((a(1)*256+a(2))*256+a(3))*256+a(4)
ImageNum = ((a(5)*256+a(6))*256+a(7))*256+a(8)
%�������ᵽ����ҳ����������ľ�
d = zeros(1,ImageNum);
for i = 1:ImageNum
    d(i) = fread(fid,1);
end
save('test_label.mat', 'd');
% savedirectory = uigetdir('','ѡ�����ͼƬ·����');
% savepath = fullfile(savedirectory,['label']);
% fod = fopen(savepath, 'w');
% for i=1:ImageNum
%     b = fread(fid,1);   %fread����Ҳ�Ǻ��ĵĺ���֮һ��b��¼����һ��ͼ�����ݴ���ע�����ﻹ�Ǹ������ǿ������κζ��ߵġ�
%     fwrite(fod, b);
% end
fclose(fid);
% fclose(fod);