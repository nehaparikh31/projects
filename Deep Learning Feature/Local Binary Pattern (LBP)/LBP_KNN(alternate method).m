close all;
clear all;
clc;

%Testing data
folder = './DogCat_KNN/Testing/';
files = dir(fullfile(folder, '*.jpg')); %dir = directory
feats = zeros(length(files), 256);


%For Cat
for i = 1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folder filename]);
    img = imresize(img,[320,480]);
    feat = lbp(img);
    feats(i,:) = feat;
end

%Cat 70% Accuracy (5,8,10)
%Dog 40% Accuracy (3, 5, 9, 10)


query = imread('./DogCat/Training/Dog/3.jpg');
query = imresize(query,[320,480]);
feat = lbp(query);
dists = distChiSq(feat,feats);
[val, idx] = min(dists);

img_result = imread([folder files(idx,1).name]);
img_result = imresize(img_result,[320,480]);
subplot(1,2,1); imshow(query); title('Query');
subplot(1,2,2); imshow(img_result); title('Result');