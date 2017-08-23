function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


%	Sigmoid function for given X*theta
XTheta = X*theta;		%	[m x (n+1)] x [(n+1) x 1] = m x 1
step1 = e.^-XTheta;		%	m x 1
step2 = 1. + step1;		%	denominator of sigmoid
h = 1./step2;			%	sigmoid function result - size (m x 1)

logH = log(h);			% size (mx1)
logOneMinusH = log(1. - h);		% size (mx1)

J = (-1/m)*((y'*logH) + ((1. - y)'*logOneMinusH));	% number

grad = (1/m)*(X')*(h - y);	% size [(n+1)x1] - same as theta

% =============================================================

end
