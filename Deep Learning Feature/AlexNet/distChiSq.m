function D = distChiSq(X, Y) %where X and Y are matrices.
m = size(X, 1); %size gets number of rows
n = size(Y, 1);

mOnes = ones(1, m);
D = zeros(m, n);

for i = 1 : n
    yi = Y(i, :);   % yi is sample of Y (instance) 
    yiRep = yi(mOnes, :);   %yirep is a matrix because mOnes is a vector with m values. (mOnes , :) will clone m n times.
    s = yiRep + X;  %addition of 2 matrix.
    d = yiRep - X;
    D(:, i) = sum(d .^ 2 ./ (s+eps), 2);    %2 because it is second determinant
end
D = D / 2;