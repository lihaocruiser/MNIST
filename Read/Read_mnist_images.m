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
a = fread(fid,16,'uint8'); %������Ҫ˵�����ǣ�����ǰʮ��λ��˵����Ϣ���������ᵽ���Ǹ���ҳ���Կ���������һλ����ʲô���塣����a������ȡ����Щ��Ϣ������¼�������������Ľ�������ȶ�����
MagicNum = ((a(1)*256+a(2))*256+a(3))*256+a(4)
ImageNum = ((a(5)*256+a(6))*256+a(7))*256+a(8)
ImageRow = ((a(9)*256+a(10))*256+a(11))*256+a(12)
ImageCol = ((a(13)*256+a(14))*256+a(15))*256+a(16)
%�������ᵽ����ҳ����������ľ�
savedirectory = uigetdir('','ѡ�����ͼƬ·����');
h_w = waitbar(0,'���Ժ򣬴�����>>');
for i=1:ImageNum
    b = fread(fid,ImageRow*ImageCol,'uint8');   %fread����Ҳ�Ǻ��ĵĺ���֮һ��b��¼����һ��ͼ�����ݴ���ע�����ﻹ�Ǹ������ǿ������κζ��ߵġ�
    c = reshape(b,[ImageRow ImageCol]); %�������ˣ�reshape���¹��ɾ������ڰѴ�ת�������ˡ�������֪ͼƬ���Ǿ�������reshape�����ĻҶȾ�����Ǹ���д���ֵľ����ˡ�
    d = c'; %ת��һ�£���Ϊc�������Ǻ��ŵġ�����
    e = 255-d; %���ݻҶ����ۣ�0�Ǻ�ɫ��255�ǰ�ɫ��Ϊ��Ū�ɰ׵׺��־ͼ�����e
    e = uint8(e);
    savepath = fullfile(savedirectory,['train' num2str(i) '.bmp']);
    imwrite(e,savepath,'bmp'); %�����imwriteд��ͼƬ
    waitbar(i/ImageNum);
end
fclose(fid);
close(h_w);