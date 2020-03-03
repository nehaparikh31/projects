close all;
clear all;
clc;


% For Cat
folder = './DogCat_KNN/Testing/';
files = dir(fullfile(folder, '*.jpg'));
feats = zeros(length(files), 3072);
feat = zeros(1, 3072);

for i  = 1 : length(files)
    %disp(i);
    filename = files(i, 1).name;
    img = imread([folder, filename]);
    img = imresize(img, [32, 32]);
%   img = rgb2gray(img);
    feat = img(:);
    feat = double(feat');
    feats(i, :) = feat;
end

%extract the features from query1 image
query = imread('./DogCat/Training/Dog/1.jpg');
query = imresize(query, [32, 32]);
% img = rgb2gray(img);
feat = query(:);
feat = double(feat');

dist = distance(feat, feats);
[val, idx] = min(dist);
% idx = double(idx);

k = 3;
for k = 1 : k
    [m,idx] = min(dist);
    dist(bsxfun(@eq,dist,m))= Inf;
    
    img_result = imread([folder files(idx,1).name]);
    img_result = imresize(img_result, [32, 32]);
    subplot(1, 2, 1); imshow(query); title('Query');
    subplot(1, 2, 2); imshow(img_result); title('Result');
end

%----------------------------------------------------------------------------------------------------------------
% % For Dog
% folder = './DogCat/Testing/Dog/';
% files = dir(fullfile(folder, '*.jpg'));
% feats = zeros(length(files), 3072);
% feat = zeros(1, 3072);
% 
% for i  = 1 : length(files)
%     %disp(i);
%     filename = files(i, 1).name;
%     img = imread([folder, filename]);
%     img = imresize(img, [32, 32]);
% %   img = rgb2gray(img);
%     feat = img(:);
%     feat = double(feat');
%     feats(i, :) = feat;
% end
% 
% %extract the features from query1 image
% query = imread('./DogCat/Training/Dog/5.jpg');
% query = imresize(query, [32, 32]);
% % img = rgb2gray(img);
% feat = query(:);
% feat = double(feat');
% %feat = TinyImage(img);;
% 
% %dist = sqrt(sumsqr(minus(single(feat),single(feats))));
% % dist = minus(double(feat), double(feats));
% % [val, idx] = min(dist);
% % dist = distance(feat, feats, 1);
% % [idx, val] = min(dist);
% % x = minus(feats, feat);
% % % disp(x);
% % y = sum(x);
% % % disp(y);
% % 
% % dist(:,i) = sqrt(sum(minus(feat,feats)).^2);
% % dist = dist /2;
% % % dist = knnsearch(feat, feats, 'K', 1, 'distance', 'euclidean');
% % % dist = dist';
% 
% dist = distance(feat, feats);
% [val, idx] = min(dist);
% % idx = double(idx);
% 
% % k = 5;
% % v = zeros(k,1);
% % 
% % for i = 1 : k  
% %     [v(i),index] = min(dist);
% %   % remove for the next iteration the last smallest value:
% %   dist(i) = [];
% % end
% 
% k = 3;
% for k = 1 : k
%     [m,idx] = min(dists);
%     dist(bsxfun(@eq,dist,m))= Inf;
% img_result = imread([folder files(idx,1).name]);
% img_result = imresize(img_result, [32, 32]);
% subplot(1, 2, 1); imshow(query); title('Query');
% subplot(1, 2, 2); imshow(img_result); title('Result');
%end