function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% 	theta size [2x1], X already has a bias unit column
h = X*theta;									% 	[12x2]*[2x1] ==> h is of size [12x1] or [mx1]
J = (1/(2*m))*(sum((h - y).^2));
J = J + ((lambda/(2*m))*sum([theta(2:end)].^2));

grad = X'*(X*theta - y);						%  	[(n+1)xm]*[mx1] = [(n+1)x1] - the matrix computation ends up doing the SUM also
grad = grad + lambda*[0;theta(2:end)];			% 	Regularlization for Theta0 is zero - grad is size [(n+1)x1] - one element per value of theta
grad = grad/m;


% =========================================================================

grad = grad(:);

end
