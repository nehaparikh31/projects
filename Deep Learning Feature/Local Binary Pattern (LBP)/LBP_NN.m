close all;
clear all;
clc;

%Preparing training data
disp('Preparing training data');
folderCat = './DogCat/Training/Cat/';
folderDog = './DogCat/Training/Dog/';

filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));

feats = zeros(length(filesCat) + length(filesDog), 256);
labels = zeros(length(filesCat) + length(filesDog),1);

% Reading the data of Cat class
for i = 1:length(filesCat)
    disp(i);
    filename = filesCat(i,1).name;
    img = imread([folderCat filename]);
    img = imresize(img,[256,256]);
    feat = lbp(img);
    feats(i,:) = feat;
    labels(i) = 0;
end


%Reading the data of Dog class
for i = 1:length(filesDog)
    disp(i);
    filename = filesDog(i,1).name;
    img = imread([folderDog filename]);
    img = imresize(img,[256,256]);
    feat = lbp(img);
    feats(i + length(filesCat),:) = feat;
    labels(i + length(filesCat)) = 1;
end


%Preparing testing data
disp('Preparing testing data');
folderCat = './DogCat/Testing/Cat/';
folderDog = './DogCat/Testing/Dog/';

filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));

featsTest = zeros(length(filesCat) + length(filesDog), 256);
groundtruthLabel = zeros(length(filesCat) + length(filesDog), 1);
predictedLabel = zeros(length(filesCat) + length(filesDog), 1);


%Reading the testing data of Cat class
for i = 1:length(filesCat)
    disp(i);
    filename = filesCat(i,1).name;
    img = imread([folderCat filename]);
    img = imresize(img,[256,256]);
    feat = lbp(img);
    featsTest(i,:) = feat;
    groundtruthLabel(i) = 0;
end

%Reading the testing data of Dog class
for i = 1:length(filesDog)
    disp(i);
    filename = filesDog(i,1).name;
    img = imread([folderDog filename]);
    img = imresize(img,[256,256]);
    feat = lbp(img);
    featsTest(i + length(filesCat),:) = feat;
    groundtruthLabel(i + length(filesCat)) = 1;
end

disp('Performing training');
net = feedforwardnet(10);
net = train(net,feats',labels');


disp('Performing testing');
predictedLabel = net(featsTest');
accurateClassification = 0;


for i = 1:size(featsTest,1)
    if predictedLabel(i)>= 0.5
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