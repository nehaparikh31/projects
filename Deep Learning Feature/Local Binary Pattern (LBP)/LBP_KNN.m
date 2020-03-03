close all;
clear all;
clc;

disp('Preparing Training data');
folderCat = './DogCat/Training/Cat/';
folderDog = './DogCat/Training/Dog/';

filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));

feats = zeros(length(filesCat) + length(filesDog), 256);
labels = zeros(length(filesCat) + length(filesDog), 1);

%for cat - training
for i = 1:length(filesCat)
    disp(i);
    filename = filesCat(i,1).name;
    img = imread([folderCat filename]);
    img = imresize(img,[256,256]);
    
    feat = lbp(img);
    labels(i) = 1;
end

% for dog - training
for i = 1:length(filesDog)
    disp(i);
    filename = filesDog(i,1).name;
    img = imread([folderDog filename]);
    img = imresize(img,[256,256]);
    feat = lbp(img);
   
    feats(i + length(filesCat),:) = feat;
    labels(i + length(filesCat)) = 2;
end

disp('Preparing Testing data');
folderCat = './DogCat/Testing/Cat/';
folderDog = './DogCat/Testing/Dog/';
filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));

featsTest = zeros(length(filesCat) + length(filesDog), 256);
groundtruthLabel = zeros(length(filesCat) + length(filesDog), 1);
predictedLabel = zeros(length(filesCat) + length(filesDog), 1);

for i = 1 : length(filesCat)
    disp(i);
    filename = filesCat(i,1).name;
    img = imread([folderCat filename]);
    img = imresize(img,[256,256]);
    feat = lbp(img);
   
    featsTest(i,:) = feat;
    groundtruthLabel(i) = 1;
end

for i = 1 : length(filesDog)
    disp(i);
    filename = filesDog(i,1).name;
    img = imread([folderDog filename]);
    img = imresize(img,[256,256]);
    feat = lbp(img);
    
    featsTest(i + length(filesCat), :) = feat;
    groundtruthLabel(i + length(filesCat)) = 2;
end
disp('Performing Testing');
accurateClassification = 0;

for i = 1 : size(featsTest,1)
    feat = featsTest(i, :);
    dists = distChiSq(feat, feats);
    [val, idx] = sort(dists);
   
    prediction = 0;
    
    k = 7;
    
    for n = 1 : k
        if(groundtruthLabel(i) == labels(idx(n)))
            prediction = prediction + 1; 
        end
    end
    
    prediction = prediction / k;
    
    if(prediction > 0.5)
        accurateClassification = accurateClassification + 1;
    end
end

accuracy = accurateClassification/length(groundtruthLabel);
disp(['The accuracy:' num2str(accuracy * 100) '%']);
