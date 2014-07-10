function [error, error_grad] = regressionObjVal(w, X, y, lambda)

% compute squared error (scalar) and gradient of squared error with respect
% to w (vector) for the given data X and y and the regularization parameter
% lambda

% computing the error using the objective function given for Problem 2
error = computeRegularizedSquaredLoss(X,y,lambda,w); 

error_grad = zeros(size(w));

% computing gradient by the result we got by differentiating the objective
% function wrt d(wj)
for j=1:size(w)
    error_grad(j) =   sum(2.*(X*w - y) .* X(:,j))/size(X,1) + w(j)*lambda ;
end

