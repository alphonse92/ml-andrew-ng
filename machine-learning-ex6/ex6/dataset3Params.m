function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;


% for the next values of sigma and C
% c_vector = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
% sigma_vector = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';

%  we got:
%
%    errors = [0.565000   0.060000   0.045000   0.145000   0.180000   0.180000   0.180000   0.180000
%    0.565000   0.060000   0.045000   0.140000   0.180000   0.180000   0.180000   0.180000
%    0.565000   0.060000   0.045000   0.080000   0.170000   0.185000   0.180000   0.185000
%    0.565000   0.060000   0.035000   0.070000   0.105000   0.180000   0.180000   0.180000
%    0.565000   0.065000   0.030000   0.070000   0.080000   0.160000   0.185000   0.180000
%    0.565000   0.080000   0.035000   0.075000   0.080000   0.100000   0.185000   0.180000
%    0.565000   0.080000   0.070000   0.070000   0.085000   0.075000   0.170000   0.185000
%    0.565000   0.080000   0.060000   0.070000   0.105000   0.070000   0.105000   0.185000]
%  less errors with C = 1 and sigma=0.1 because  0.030000

c_vector = [1]';
sigma_vector = [0.1]';

% matrix of predictions
size_of_c_vector = size(c_vector,1);
size_of_sigma_vector = size(sigma_vector,1);
errors=zeros(size_of_c_vector,size_of_sigma_vector);

for iCVector = 1 : size_of_c_vector
    for iSigmaVector = 1 : size_of_sigma_vector;
        current_c = c_vector(iCVector);
        current_sigma = sigma_vector(iSigmaVector);
        model= svmTrain(X, y, current_c, @(x1, x2) gaussianKernel(x1, x2, current_sigma)); 
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        errors(iCVector,iSigmaVector) = error;
    end;
end;


[minval, posSigma] = min(min(errors,[],1));
[minval, posC] = min(min(errors,[],2));

C = c_vector(posC)
sigma = sigma_vector(posSigma)

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% =========================================================================

end
