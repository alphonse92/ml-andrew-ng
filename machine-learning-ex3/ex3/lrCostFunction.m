function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
thetaWithoutTetha1 = theta;
thetaWithoutTetha1(1)=0;
XwithoutOnes = X(:,[2,size(X,2)]);
XfirstColumn = X(:,[1]);
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

predictions = X*theta; # mxn * nx1 = mx1
z = sigmoid(predictions); # still mx1
left = -y .* log(z); # -(mx1) .* log(mx1)
right = (1 .- y) .* log(1 - z); # still mx1 , sum(mx1) = 1x1
costWithoutReg =  (1/m) * sum(left.-right); 
costFactorReg = (lambda/(2*m)) * sum(thetaWithoutTetha1 .^2) ; 
J = costWithoutReg + costFactorReg ;


grad = zeros(size(theta));
errors = z .- y; # mx1
errorsWithX = X .* errors ; # mxn .* mx1 = mxn
gradWithNotReg = (1/m) .* sum(errorsWithX); # sum(nx1)
gradFactorReg = (lambda/m) .* thetaWithoutTetha1;
grad = gradWithNotReg + gradFactorReg';
# the original exercise told to us to return grad(:), it means this function should
# return an vector
grad = grad';
end
