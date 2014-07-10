% load the data
load diabetes;
x_train_i = [ones(size(x_train,1),1) x_train];
x_test_i = [ones(size(x_test,1),1) x_test];
%%% FILL CODE FOR PROBLEM 1 %%%
% linear regression without intercept

weight = learnOLERegression(x_train,y_train);
error_train = computeRSE(x_train,weight,y_train);
error_test = computeRSE(x_test,weight,y_test);
fprintf('Linear regression without intercept\n');
fprintf('Training Error:= %f\n',error_train);
fprintf('Testing Error:= %f\n',error_test);

% linear regression with intercept

weight = learnOLERegression(x_train_i,y_train);
error_train = computeRSE(x_train_i,weight,y_train);
error_test = computeRSE(x_test_i,weight,y_test);
fprintf('Linear regression with intercept\n');
fprintf('Training Error:= %f\n',error_train);
fprintf('Testing Error:= %f\n',error_test);

%%% END PROBLEM 1 CODE %%%

%%% FILL CODE FOR PROBLEM 2 %%%
% ridge regression using least squares - minimization
lambdas = 0:0.00001:0.001;
train_errors = zeros(length(lambdas),1);
test_errors = zeros(length(lambdas),1);
min_training_error = -1;

for i = 1:length(lambdas)
    lambda = lambdas(i);
    % fill code here for prediction and computing errors
    
    weight = learnRidgeRegression(x_train_i,y_train,lambda);
    train_errors(i,1) = computeRSE(x_train_i,weight,y_train); %computeRegularizedSquaredLoss(x_train_i,y_train,lambda,weight);
    
    test_errors(i,1)= computeRSE(x_test_i,weight,y_test);
    if(min_training_error == -1)
        min_training_error = test_errors(i,1);
        lambda_optimal = lambda;
    elseif(train_errors(i,1)<min_training_error)
        min_training_error = test_errors(i,1);
        lambda_optimal = lambda;
    end
    
end
figure;
plot([train_errors test_errors]);
legend('Training Error','Testing Error');
set(gca,'XTickLabel',sprintf('%0.5f|',lambdas));
ylabel('error');
xlabel('lambda');

%%% END PROBLEM 2 CODE %%%

%%% BEGIN PROBLEM 3 CODE
% ridge regression using gradient descent - see handouts (lecture 21 p5) or
% http://cs229.stanford.edu/notes/cs229-notes1.pdf (page 11)
initialWeights = zeros(65,1);
% set the maximum number of iteration in conjugate gradient descent
options = optimset('MaxIter', 500);

% define the objective function
lambdas = 0:0.00001:0.001;
train_errors = zeros(length(lambdas),1);
test_errors = zeros(length(lambdas),1);
lambda_optimal_gd = 0;
min_error_gd = -1;
% run ridge regression training with fmincg
for i = 1:length(lambdas)
    lambda = lambdas(i);
    objFunction = @(params) regressionObjVal(params, x_train_i, y_train, lambda);
    w = fmincg(objFunction, initialWeights, options);
    % fill code here for prediction and computing errors
    
    train_errors(i,1) = computeRSE(x_train_i,w,y_train); 
    test_errors(i,1)= computeRSE(x_test_i,w,y_test);
    
    % looking for optimal lambda
    if(min_error_gd == -1)      
        min_error_gd = test_errors(i,1);
    elseif(test_errors(i,1) < min_error_gd)
        min_error_gd = test_errors(i,1);
        lambda_optimal_gd = lambda;
    end
end
figure;
plot([train_errors test_errors]);
legend('Training Error','Testing Error');
set(gca,'XTickLabel',sprintf('%0.5f|',lambdas));
ylabel('error');
xlabel('lambda');

%%% END PROBLEM 3 CODE

%%% BEGIN  PROBLEM 4 CODE
% using variable number 3 only
x_train = x_train(:,3);
x_test = x_test(:,3);
train_errors = zeros(7,1);
test_errors = zeros(7,1);

% no regularization
lambda = 0;
for d = 0:6
    x_train_n = mapNonLinear(x_train,d);
    x_test_n = mapNonLinear(x_test,d);
    w = learnRidgeRegression(x_train_n,y_train,lambda);
    % fill code here for prediction and computing errors
    
    train_errors(d+1,1) = computeRSE(x_train_n,w,y_train);
    test_errors(d+1,1) = computeRSE(x_test_n,w,y_test);
    %fprintf('train error %f\n',train_errors);
    %fprintf('test error %f\n',test_errors);
end

figure;
plot([train_errors test_errors]);
legend('Training Error','Testing Error');
set(gca,'XTickLabel',{'0','1','2','3','4','5','6'});
xlabel('d');
ylabel('error');

% optimal regularization
lambda = lambda_optimal; % from part 2
for d = 0:6
    x_train_n = mapNonLinear(x_train,d);
    x_test_n = mapNonLinear(x_test,d);
    w = learnRidgeRegression(x_train_n,y_train,lambda);
    % fill code here for prediction and computing errors
    
    train_errors(d+1,1) = computeRSE(x_train_n,w,y_train);
    test_errors(d+1,1) = computeRSE(x_test_n,w,y_test);
end
figure;
plot([train_errors test_errors]);
legend('Training Error','Testing Error');
set(gca,'XTickLabel',{'0','1','2','3','4','5','6'});
xlabel('d');
ylabel('error');

fprintf('Analytically computed optimal lambda := %f\n',lambda_optimal);
fprintf('Optimal lambda for gradient descent := %f\n',lambda_optimal);