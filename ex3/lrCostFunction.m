function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


%	Sigmoid function for given X*theta
XTheta = X*theta;		%	[m x (n+1)] x [(n+1) x 1] = m x 1
step1 = e.^-XTheta;		%	m x 1
step2 = 1. + step1;		%	denominator of sigmoid
h = 1./step2;			%	sigmoid function result - size (m x 1)

logH = log(h);			% size (mx1)
logOneMinusH = log(1. - h);		% size (mx1)

J = (-1/m)*((y'*logH) + ((1. - y)'*logOneMinusH));	% (COST FUNCTION - number)

%%	Add regularization function
theta2 = theta(2:end);		%% 	ignore theta(0)
foo = sum(theta2.^2);		%%	sum of theta elements theta(1) to theta(n)
J = J + ((lambda/(2*m))*foo);	%%	regularization function added to J


grad = ((1/m)*(X')*(h - y)) + ((lambda/m)*theta);	% size [(n+1)x1] - same as theta
grad(1) = grad(1) - ((lambda/m)*theta(1));			% don't regularize theta(0)


% =============================================================

% grad = grad(:);

end
