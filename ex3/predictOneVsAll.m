function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);						% size m (5000)
num_labels = size(all_theta, 1);	% size K (10)

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);			% size (mx1)

% Add ones to the X data matrix
X = [ones(m, 1) X];					% size [mx(n+1)]

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

z = X*all_theta';		% size (mxk) --> (5000x10) size........ 	One row per classification (1-10)
h = sigmoid(z);			% size (mxk) --> (5000x10) size

% 	for each row, find max value and its position, size of p and max_val is (5000x1) - one per row, or per classification
[max_val p] = max(h, [], 2);





% =========================================================================


end
