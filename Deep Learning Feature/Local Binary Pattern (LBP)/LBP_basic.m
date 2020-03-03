%Using LBP
clear all;
close all;
clc;


%Reading training data of cat and dog
disp('Preparing training data');

folderCat = './DogCat/Training/Cat/';
folderDog = './DogCat/Training/Dog/';

filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));

featsCat = zeros(length(filesCat), 256);
featsDog = zeros(length(filesDog), 256);


%Training data of Cat
for i  = 1 : length(filesCat)
    %disp(i);
    filename = filesCat(i, 1).name;
    img = imread([folderCat, filename]);
    %img = imresize(img, [320, 480]);
    feat = lbp(img);
    title('Training Data of Cat');
    figure, bar(feat);
end


%Training data of Dog
for i  = 1 : length(filesDog)
    %disp(i);
    filename = filesDog(i, 1).name;
    img = imread([folderDog, filename]);
    %img = imresize(img, [320, 480]);
    feat = lbp(img);
    title('Training Data of Dog');
    figure, bar(feat);
end


%Testing Data
disp('Preparing testing data');

folder_TestCat = './DogCat/Training/Cat/';
folder_TestDog = './DogCat/Training/Dog/';

files_TestCat = dir(fullfile(folder_TestCat, '*.jpg'));
files_TestDog = dir(fullfile(folder_TestDog, '*.jpg'));

feats_TestCat = zeros(length(files_TestCat), 256);
feats_TestDog = zeros(length(files_TestDog), 256);


%Testing data of Cat
for i  = 1 : length(files_TestCat)
    %disp(i);
    filename = files_TestCat(i, 1).name;
    img = imread([folder_TestCat, filename]);
    %img = imresize(img, [320, 480]);
    feat = lbp(img);
    title('Testing Data of Cat');
    figure, bar(feat);
end

%Testing data of Dog
for i  = 1 : length(files_TestDog)
    %disp(i);
    filename = files_TestDog(i, 1).name;
    img = imread([folder_TestDog, filename]);
    %img = imresize(img, [320, 480]);
    feat = lbp(img);
    title('Testing Data of Dog');
    figure, bar(feat);
end