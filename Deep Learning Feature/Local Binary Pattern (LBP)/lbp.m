% LBP function is used to extract local binary patterns from an input image

function feat = lbp(img)
 %Step 1: Convert the 3 channel image to gray scale.
img_gray = rgb2gray(img);  

%Step 2: Get the size of the image. The size of original image and gray scale image are identical.
[height, width] = size(img_gray);   
feat = zeros(1, 256);   %row vector with all zeros in 1 row and having 256 columns.

%Step 3: Going through all pixels and considering neighbourhood as 3 X 3.
for i = 2 : height - 1  % reason written in notes.
    for j = 2 : width - 1
        %Step 4: Check for neighbours. WE can get a minor matrix from the original gray scaled matrix.
        neighbors = img_gray(i-1:i+1,j-1:j+1);  % neighbors are sub-matrix
        bits = double(neighbors(:));    %: means we are concatenating all the values.% bits is the values.
        threshold = bits(5);    %center value (1-9 center bit is 5)
        bits(5) = [];
        bits = bits - threshold;    %bits is a vector and threshold is a scaler. %%See notes for detailed explanation
        bits = sign(bits);
        bits(bits < 0) = 0;
        
        %convert bit to byte value
        byte = sum(bits .* 2 .^ (length(bits)- 1 : -1 : 0)');   %%detailed explanation in notes.
        
        %Step 5: Update the Histogram
        feat(byte + 1) = feat(byte + 1) + 1;
        %feat is a vector of 256 values. while byte has a value from 0 to
        %255 and in matlab index starts from 1 so we wont find the final
        %value. So we add 1 to feat.
        
    end
    %If we have image with 1000 or 5000 pixels and then the histogram
    %will be different everytime. So we need to perform normalization 
    feat = feat ./ sum(feat);    
end
end
