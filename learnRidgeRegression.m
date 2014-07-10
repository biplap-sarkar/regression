function w = learnRidgeRegression(X,y,lambda)

% Implement ridge regression training here
% Inputs:
% X = N x D
% y = N x 1
% lambda = scalar
% Output:
% w = D x 1
n = size(X,1);
d = size(X,2);
w = inv((lambda*eye(d)*n) +  (X' * X))* X' * y;
