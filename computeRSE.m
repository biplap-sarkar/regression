% computing root squared error as specified in Problem 1
function rse = computeRSE(X,w,y)
rse = sqrt(sum((y - X*w).^2));

