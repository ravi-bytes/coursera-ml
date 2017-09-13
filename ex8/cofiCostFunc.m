function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


#{
	R is rating vector containg 1 where rating has been provided by user "j" for movie "i".
	0 everywhere else.
#}


%===========  NON REGULARIZED COST and GRADIENT ==================%
J = ((X*Theta') - Y); 				% 	Cost function computation
J = J.*R;							% 	Changes cost to 0 where movie "i" has not been rated by user "j"
J = J.^2;							% 	Square elements
J = 0.5*(sum(sum(J)));				% 	Sum over all movies for all users * 0.5


A = ((X*Theta') - Y);				% 	(num_movies x num_users)
A = A.*R;							% 	(num_movies x num_users) - non rated elements are zero
X_grad = A*Theta;					% 	(num_movies x num_features)
Theta_grad = A'*X;					% 	(num_users x num_features)

%===========  REGULARIZED COST and GRADIENT ==================%

J = sum(sum(J));
J1 = 0.5*lambda*sum(sum(Theta.^2));
J2 = 0.5*lambda*sum(sum(X.^2));
J = J + J1 + J2;

X_grad = X_grad + (lambda*X);

Theta_grad = Theta_grad + (lambda*Theta);

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
