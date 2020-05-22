function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);
m = size(X,1);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

Z = zeros(m,K);
for i= 1:m
    u_reduce = U(:,1:K); % nxn => nxk
    x = X(i,:); % 1xn
    projection = x*u_reduce; % 1xn * nxk = 1xk
    Z(i,:) = x * u_reduce; 
end;

% =============================================================

end
