close all;
clear all;
clc;

%Add MatConvNet 
run './matconvnet/matlab/vl_setupnn'

%Load the pre-trained CNN
net = load('imagenet-caffe-alex.mat');

disp('Preparing training data');
folderCat = './Updated_DogCat/Training/Cat/';
folderDog = './Updated_DogCat/Training/Dog/';

filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));

feats = zeros(length(filesCat) + length(filesDog), 4096);
labels = zeros(length(filesCat) + length(filesDog),1);

for i = 1:length(filesCat)
    disp(i);
    filename = filesCat(i, 1).name;
    im = imread([folderCat filename]);
    im_ = single(im);
    im_ = imresize(im_, net.meta.normalization.imageSize(1 : 2)) ;

    for j = 1 : 3
        im_(:, :, j) = im_(:, :, j) - net.meta.normalization.averageImage(j);
    end
    % run the CNN
    res = vl_simplenn(net, im_) ;
    
    feat = res(18).x;   %fc7 is layer 18 in net.layers
    feat = feat(:)';
    feats(i,:) = feat;
    
    labels(i) = 1;
end

for i = 1 : length(filesDog)
    disp(i);
    filename = filesDog(i, 1).name;
    im = imread([folderDog filename]);
    im_ = single(im);
    im_ = imresize(im_, net.meta.normalization.imageSize(1 : 2)) ;

    for j = 1 : 3
        im_(:, :, j) = im_(:, :, j) - net.meta.normalization.averageImage(j);
    end
    
    % run the CNN
    res = vl_simplenn(net, im_) ;
    feat = res(18).x;
    feat = feat(:)';
    
    feats(i + length(filesCat),:) = feat;
    labels(i + length(filesCat)) = 2;
end


% Preparing testing data
disp('Preparing testing data');

folderCat = './Updated_DogCat/Testing/Cat/';
folderDog = './Updated_DogCat/Testing/Dog/';

filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));

featsTest = zeros(length(filesCat) + length(filesDog), 4096);

groundtruthLabel = zeros(length(filesCat) + length(filesDog), 1);
predictedLabel = zeros(length(filesCat) + length(filesDog), 1);

%Testing on Cat
for i = 1 : length(filesCat)
    disp(i);
    filename = filesCat(i, 1).name;
    im = imread([folderCat filename]);
    im_ = single(im);
    im_ = imresize(im_, net.meta.normalization.imageSize(1 : 2)) ;

    for j = 1 : 3
        im_(:, :, j) = im_(:, :, j) - net.meta.normalization.averageImage(j);
    end
    
    % run the CNN
    res = vl_simplenn(net, im_) ;
    feat = res(18).x;
    feat = feat(:)';
    featsTest(i,:) = feat;
    
    groundtruthLabel(i) = 1;
end


%Testing on Dog
for i = 1 : length(filesDog)
    disp(i);
    filename = filesDog(i, 1).name;
    im = imread([folderDog filename]);
    im_ = single(im) ;
    im_ = imresize(im_, net.meta.normalization.imageSize(1 : 2)) ;

    for j = 1 : 3
        im_(:, :, j) = im_(:, :, j) - net.meta.normalization.averageImage(j);
    end
    
    % run the CNN
    res = vl_simplenn(net, im_) ;
    feat = res(18).x;
    feat = feat(:)';
    
    featsTest(i + length(filesCat), :) = feat;
    groundtruthLabel(i + length(filesCat)) = 2;
end


disp('Performing training');
net = feedforwardnet(5);
net = train(net,feats',labels');

disp('Performing testing');
predictedLabel = net(featsTest');
accurateClassification = 0;


for i = 1:size(featsTest,1)
    if predictedLabel(i) >= 0.5
        predictedLabel(i) = 1;
    else
        predictedLabel(i) = 0;
    end

    if(predictedLabel(i) == groundtruthLabel(i))
        accurateClassification = accurateClassification + 1;
    end

end


%Compute and show the accuracy rate
accuracy = accurateClassification/length(groundtruthLabel);
disp(['The accuracy:' num2str(accuracy * 100) '%']);