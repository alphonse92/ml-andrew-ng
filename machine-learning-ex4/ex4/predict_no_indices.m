function [ a1,a2,a3,z2,z3,predictions] = predict_no_indices(Theta1, Theta2, X)

%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

a1=X; % 5000x400

z2 = [ones(m, 1) a1] * Theta1'; % 5000x401 * 401x25 = 5000x25
a2 = sigmoid(z2);  % still 5000x25

z3 = [ones(m, 1) a2] * Theta2';  % 5000x26 * 26x10 = 5000x10
a3 = sigmoid(z3); % 5000x10

predictions = a3;
end; 