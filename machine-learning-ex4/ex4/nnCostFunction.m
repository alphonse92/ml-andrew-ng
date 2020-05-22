function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, ...  % 5000x400
                                   y, ... % 5000x1
                                   lambda)

warning('off','Octave:broadcast');

%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); % 25 x 401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));  % 10 x 26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

J=0;

% cost
[ a1,a2,a3,z2,z3,predictions] =  predict_no_indices(Theta1,Theta2,X); % 5000x10
yAll = y == [1:num_labels] ;                    % 5000x10;
left = -yAll .* log(predictions);                % 5000x10 ;
right = (1 .- yAll) .* log( 1 .- predictions ); % 5000x10;
logaritmicCost = left .- right;                 % 5000x10
J = sum(sum(logaritmicCost));
J = (1/m) * J;

% regularized cost
t1 = Theta1(:,2:end); %25x400
t2 = Theta2(:,2:end);% 10x25
squaredT1 = t1.^2;
squaredT2 = t2.^2;
J = J + (lambda/(2*m)) .* (  sum(sum(squaredT1))  + sum(sum(squaredT2))   );


%backpropagation layer 3
sigma3 = a3 - yAll; % 5000x10 -  5000-10 = 5000x10;
a2_withOnes =  [ ones(size(a2,1),1) a2] ; % i need to add a extra columns filled with ones wioth the bias
Theta2_grad = sigma3' * a2_withOnes; % (5000x10)'*(5000x25) = 10x5000*5000x25 = 10*25

%backprop layer 2
sigma2 = (sigma3 * t2) .* sigmoidGradient(z2); %  (50000x10 * 10x25) .* 50000x25 = 5000x25 .* 5000x25 = 5000x25
a1_withOnes =  [ ones(size(a1,1),1) a1] ; % i need to add a extra columns filled with ones wioth the bias
Theta1_grad = sigma2' *a1_withOnes ; % 5000x25'* 5000x400   = 25x5000 * 5000x400 = 25x401

Theta1_grad = Theta1_grad ./m; % 25x400
Theta2_grad = Theta2_grad ./m; % 10x25

%reguylarize  gradients
regTheta1 = Theta1_grad + (lambda * (1 / m) * Theta1);
regTheta2 = Theta2_grad + (lambda * (1 / m) * Theta2);
%  omit the bias
Theta1_grad(:,2:end) =regTheta1(:,2:end);
Theta2_grad(:,2:end) =regTheta2(:,2:end);

% unroll parameters
grad = [Theta1_grad(:) ; Theta2_grad(:)];


warning('on','Octave:broadcast');
end
