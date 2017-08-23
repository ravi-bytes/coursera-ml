function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);	
n = size(X, 2);

%	X(mxn) ==> m=size m, n=size n

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];		% X is now (m x (n+1))

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%
%%		We have 5000 handwriting samples of numbers from 0 - 9 (0 being tagged as 10) - 500 samples of each
%%		Each set of samples is a class - we have 10 classes
%%		We are building a solution below to loop through each class and using the costfunction we have created earlier
%%		compute (n+1) values of theta for each class
%%		For each class with 500 samples, every sample is broken down into a vector of 400 elements.  We add an initial element (theta0 = 1)

for c = 1:num_labels

	initial_theta = zeros(n + 1, 1);		%%	initialize theta with zeros, (n+1 x 1) vector

	options = optimset('GradObj', 'on', 'MaxIter', 50);
	[theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);

	if (c == 1)
		theta_old = theta';	% vector with (n+1) elements
	else
		theta_old = [theta_old; theta'];	% append each sample's theta to existing - add as a new row
	end

	all_theta = theta_old;		%	size [K x n+1] - # of rows == number of samples (1-10), # of columns == (n+1) corresponds to # of thetas

end






% =========================================================================


end
