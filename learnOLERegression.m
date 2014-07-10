function w = learnOLERegression(X,y)
% Implement OLE training here
% Inputs:
% X = N x D
% y = N x 1
% Output:
% w = D x 1
w = inv(X' * X)* X' * y;
