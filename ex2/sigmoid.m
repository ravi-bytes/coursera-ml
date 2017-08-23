function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


%	Sigmoid function for given z
step1 = e.^-z;			%	size(z) matrix
step2 = 1. + step1;		%	denominator of sigmoid
g = 1./step2;		%	sigmoid function result - size(z) matrix

% =============================================================

end
