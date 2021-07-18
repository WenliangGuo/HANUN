clear all;clc;close all;
addpath('Functions\');%加载文件夹Functions中的函数
%% 三种方法得到的结果路径，以及真值图路径
% result1 = 'resultsDSR\';
% result2 = 'resultsGMR\';
% result3 = 'resultsMCA\';
 
result1 = 'D:\segmentation_nets\test\CUHK\AUPSNet\out_of_focus\results\';
result2 = 'D:\segmentation_nets\test\DUT\AUPSNet_allPP\';
result3 = 'D:\segmentation_nets\test\DUT\AUPSNet_noSE\';
result4 = 'D:\segmentation_nets\test\CUHK\BTBNet\results\';
result5 = 'D:\segmentation_nets\test\CUHK\CENet\results\';
result6 = 'D:\segmentation_nets\test\CUHK\DBDF\out_of_focus\results\';
result7 = 'D:\segmentation_nets\test\CUHK\HiFST\motion\results\';
result8 = 'D:\segmentation_nets\test\CUHK\JNB\results\';
result9 = 'D:\segmentation_nets\test\CUHK\LBP\out_of_focus\results\';
result10 = 'D:\segmentation_nets\test\CUHK\SS\out_of_focus\results\';
result11 = 'D:\segmentation_nets\test\CUHK\U2Net\out_of_focus\results\';

% mask = 'D:\segmentation_nets\test\DUT\gt\';

mask1 = 'D:\segmentation_nets\test\CUHK\AUPSNet\out_of_focus\gt\';
mask2 = 'D:\segmentation_nets\test\CUHK\AUPSNet_allpp\gt\';
mask3 = 'D:\segmentation_nets\test\CUHK\AUPSNet_noSE\gt\';
mask4 = 'D:\segmentation_nets\test\CUHK\BTBNet\gt\';
mask5 = 'D:\segmentation_nets\test\CUHK\CENet\gt\';
mask6 = 'D:\segmentation_nets\test\CUHK\DBDF\out_of_focus\gt\';
mask7 = 'D:\segmentation_nets\test\CUHK\HiFST\motion\gt\';
mask8 = 'D:\segmentation_nets\test\CUHK\JNB\gt\';
mask9 = 'D:\segmentation_nets\test\CUHK\LBP\out_of_focus\gt\';
mask10 = 'D:\segmentation_nets\test\CUHK\SS\out_of_focus\gt\';
mask11 = 'D:\segmentation_nets\test\CUHK\U2Net\out_of_focus\gt\';

%% 创建文件夹evaluation index，目的是保存PR曲线图
newFolder = 'evaluation index'; 
if ~exist(newFolder)
    mkdir(newFolder);
end

%% Evaluation index 1: evaluating MAE
%  resultSuffixDSR = '_DSR.png';
%  resultSuffixGMR = '_stage1.png';
%  resultSuffixMCA = '_MCA.png';
%  gtSuffix = '.png';

resultSuffix = '.png';
gtSuffix = '.png';

% maeAUPSNet = CalMeanMAE(result1, resultSuffix, mask1, gtSuffix);
% maeAUPSNet_allpp = CalMeanMAE(result2, resultSuffix, mask2, gtSuffix);
% maeAUPSNet_noSE = CalMeanMAE(result3, resultSuffix, mask3, gtSuffix);
% maeBTBNet = CalMeanMAE(result4, resultSuffix, mask4, gtSuffix);
% maeCENet = CalMeanMAE(result5, resultSuffix, mask5, gtSuffix);
% maeDBDF = CalMeanMAE(result6, resultSuffix, mask6, gtSuffix);
% maeHiFST = CalMeanMAE(result7, resultSuffix, mask7, gtSuffix);
% maeJNB = CalMeanMAE(result8, resultSuffix, mask8, gtSuffix);
% maeLBP = CalMeanMAE(result9, resultSuffix, mask9, gtSuffix);
% maeSS = CalMeanMAE(result10, resultSuffix, mask10, gtSuffix);
% maeU2Net = CalMeanMAE(result11, resultSuffix, mask11, gtSuffix);

%% Evaluation index 2: ploting PR curve
% [rec1, prec1] = DrawPRCurve(result1, resultSuffixDSR, mask, gtSuffix, true, true, 'r');
% hold on
% [rec2, prec2] = DrawPRCurve(result2, resultSuffixGMR, mask, gtSuffix, true, true, 'g');
% hold on
% [rec3, prec3] = DrawPRCurve(result3, resultSuffixMCA, mask, gtSuffix, true, true, 'b');
% hold off;

[rec1, prec1] = DrawPRCurve(result1, resultSuffix, mask1, gtSuffix, true, true, 'r');
hold on
% [rec2, prec2] = DrawPRCurve(result2, resultSuffix, mask2, gtSuffix, true, true, 'g');
% hold on
% [rec3, prec3] = DrawPRCurve(result3, resultSuffix, mask3, gtSuffix, true, true, 'b');
% hold on;
% [rec4, prec4] = DrawPRCurve(result4, resultSuffix, mask4, gtSuffix, true, true, 'c');
% hold on;
% [rec5, prec5] = DrawPRCurve(result5, resultSuffix, mask5, gtSuffix, true, true, 'm');
% hold on;
[rec6, prec6] = DrawPRCurve(result6, resultSuffix, mask6, gtSuffix, true, true, 'g');
hold on;
% [rec7, prec7] = DrawPRCurve(result7, resultSuffix, mask7, gtSuffix, true, true, 'b');
% hold on;
% [rec8, prec8] = DrawPRCurve(result8, resultSuffix, mask8, gtSuffix, true, true, [0.98,0.92,0.84]);
% hold on;
[rec9, prec9] = DrawPRCurve(result9, resultSuffix, mask9, gtSuffix, true, true, 'b');%[1.00,0.38,0.00]
hold on;
[rec10, prec10] = DrawPRCurve(result10, resultSuffix, mask10, gtSuffix, true, true, [0.12,0.56,1.00]);
hold on;
[rec11, prec11] = DrawPRCurve(result11, resultSuffix, mask11, gtSuffix, true, true, [0.00,1.00,1.00]);
hold off;

grid on;
box on;
xlabel('Recall');
ylabel('Precision');
% title(strcat('PR-curve','  ( ',sprintf(' MAE = %1.6f ',maeDSR),' )'));
%title('PR-curve');
%lg = legend({'AUPSNet','AUPSNet-allpp','AUPSNet-noSE','BTBNet','CENet','DBDF','JNB','LBP','SS','U2Net'});
%lg = legend({'AUPSNet','BTBNet','CENet','DBDF','JNB','LBP','SS','U2Net'});
lg = legend({'AUPSNet','DBDF','LBP','SS','U2Net'});
%lg = legend({'AUPSNet','AUPSNet-allpp','AUPSNet-noSE','U2Net'});
set(lg, 'location', 'southwest');
k=1.2;
set(gcf,'units',get(gcf,'paperunits'));
set(gcf,'paperposition',get(gcf,'position')*k);
saveas(gcf,strcat(newFolder,'\defocusing PR Curve','.png'));

