function dist = distance(X, Y)
% 
%     m = size(X,1);
%     n = size(Y,1);
% 
%     % absolute distance between all test and training data
%     dist = abs(repmat(X,1,n) - repmat(Y(:,1),m,1));
% 
% %     % indicies of nearest neighbors
% %     [~,nearest] = sort(dist,2);
% %     
% %     % k nearest
% %     nearest = nearest(:,1:k);
% % 
% %     % mode of k nearest
% %     val = reshape(Y(nearest,2),[1,80],k);
% %     dist = mode(val,2);
% %     % if mode is 1, output nearest instead
% %     dist(dist == 1) = val(dist == 1,1);


    m = size(X, 1); %size gets number of rows
    n = size(Y, 1);

    mOnes = ones(1, m);
    D = zeros(m, n);
    
    for i = 1 : n
        yi = Y(i, :);   % yi is sample of Y (instance) 
        yiRep = yi(mOnes, :);
        d = yiRep - X;
        
        D(:, i) = sum(d).^2;
        dist = sqrt(D);
    end
    