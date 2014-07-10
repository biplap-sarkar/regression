function rsl = computeRegularizedSquaredLoss(X,y,lambda,w)
n = size(X,1);
rsl = (sum((y - X*w).^2))/n + (lambda* (w' *w));