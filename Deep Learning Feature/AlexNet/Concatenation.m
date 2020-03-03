close all;
clear all;
clc;

%Add MatConvNet 
run './matconvnet/matlab/vl_setupnn'

%Load the pre-trained CNN
net = load('imagenet-caffe-alex.mat');

%Train Cat-Dog according to LBP
disp('Preparing training data');

folderCat = './FlippedImage_Dataset/Training/Cat/';
folderDog = './FlippedImage_Dataset/Training/Dog/';

filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));

feats = zeros(length(filesCat) + length(filesDog), 4352);
labels = zeros(length(filesCat) + length(filesDog),1);

disp('Training Cat Data');
for i = 1:length(filesCat)
    disp(i);
    filename = filesCat(i,1).name;
    img = imread([folderCat filename]);
    
    %using LBP
    img1 = imresize(img,[256,256]);    
    feat_LBP = lbp(img1);
    
    %Using DL
    im_ = single(img);
    im_ = imresize(im_, net.meta.normalization.imageSize(1 : 2)) ;

    for j = 1 : 3
        im_(:, :, j) = im_(:, :, j) - net.meta.normalization.averageImage(j);
    end
    
    % run the CNN
    res = vl_simplenn(net, im_) ;
    
    feat_DL = res(18).x;   %fc7 is layer 18 in net.layers
    feat_DL = feat_DL(:)';
    
    feat = [feat_LBP, feat_DL];
    feats(i,:) = feat;
    
    
    labels(i) = 1;
end

disp('Training Dog Data');
for i = 1:length(filesDog)
    disp(i);
    filename = filesDog(i,1).name;
    img = imread([folderDog filename]);
    
    %using LBP
    img1 = imresize(img,[256,256]);    
    feat_LBP = lbp(img1);
    
    %Using DL
    im_ = single(img);
    im_ = imresize(im_, net.meta.normalization.imageSize(1 : 2)) ;

    for j = 1 : 3
        im_(:, :, j) = im_(:, :, j) - net.meta.normalization.averageImage(j);
    end
    
    % run the CNN
    res = vl_simplenn(net, im_) ;
    
    feat_DL = res(18).x;   %fc7 is layer 18 in net.layers
    feat_DL = feat_DL(:)';
    
    feat = [feat_LBP, feat_DL];
    
    feats(i + length(filesCat),:) = feat;
    labels(i + length(filesCat)) = 2;
end

disp('Preparing testing data');
folderCat = './FlippedImage_Dataset/Testing/Cat/';
folderDog = './FlippedImage_Dataset/Testing/Dog/';

filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));

featsTest = zeros(length(filesCat) + length(filesDog), 4352);

groundtruthLabel = zeros(length(filesCat) + length(filesDog), 1);
predictedLabel = zeros(length(filesCat) + length(filesDog), 1);

disp('Testing Cat Data')
for i = 1:length(filesCat)
    disp(i);
    filename = filesCat(i,1).name;
    img = imread([folderCat filename]);
    
    %Using LBP
    img1 = imresize(img,[256,256]);    
    feat_LBP = lbp(img1);
    
    %Using DL
    im_ = single(img);
    im_ = imresize(im_, net.meta.normalization.imageSize(1 : 2)) ;

    for j = 1 : 3
        im_(:, :, j) = im_(:, :, j) - net.meta.normalization.averageImage(j);
    end
    
    % run the CNN
    res = vl_simplenn(net, im_) ;
    
    feat_DL = res(18).x;   %fc7 is layer 18 in net.layers
    feat_DL = feat_DL(:)';
    
    feat = [feat_LBP, feat_DL];
    
    featsTest(i,:) = feat;
    groundtruthLabel(i) = 1;
end

disp('Testing Dog Data')
for i = 1:length(filesDog)
    disp(i);
    filename = filesDog(i,1).name;
    img = imread([folderDog filename]);
    
    %Using LBP
    img1 = imresize(img,[256,256]);    
    feat_LBP = lbp(img1);
    
    %Using DL
    im_ = single(img);
    im_ = imresize(im_, net.meta.normalization.imageSize(1 : 2)) ;

    for j = 1 : 3
        im_(:, :, j) = im_(:, :, j) - net.meta.normalization.averageImage(j);
    end
    
    % run the CNN
    res = vl_simplenn(net, im_) ;
    
    feat_DL = res(18).x;   %fc7 is layer 18 in net.layers
    feat_DL = feat_DL(:)';
    
    feat = [feat_LBP, feat_DL];
    
    featsTest(i + length(filesCat),:) = feat;
    groundtruthLabel(i + length(filesCat)) = 2;
end

disp('Performing testing');
accurateClassification = 0;

for i = 1:size(featsTest,1)
    feat = featsTest(i,:);
    dists = distChiSq(feat,feats);
    [val, idx] = sort(dists);
    
    cat_count = 0; 
    dog_count =0 ;
    k = 3;
    
    for n = 1 : k
        if(labels(idx(n)) == 1)
            cat_count = cat_count + 1;
        elseif(labels(idx(n)) == 2)
            dog_count = dog_count + 1;    
        end
    end
    
    if(cat_count > dog_count)
        predictedLabel(i) = 1;
    else
        predictedLabel(i) = 2;
    end
    
    if(predictedLabel(i) == groundtruthLabel(i))
        accurateClassification = accurateClassification + 1;
    end
end

accuracy = accurateClassification/length(groundtruthLabel);
disp(['The accuracy:' num2str(accuracy * 100) '%']);
