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
%%	Prediction means calculating h(theta) for every class (a class here is a 500-count sample set of handwritten number from 0-9)
%%	All we are doing here is calculating (sum[theta0*x0 + theta1*x1 + ....... + thetan*xn]) for every class
%%	all_theta has one row per class - each row containing (n+1) elements
%%	X is the original data - 5000 (m) records, each with (n+1) elements
%%  Multiplying the two gives us a 5000x10 matrix
%%		Each row of the matrix represents one of the sample pictures
%%		Each column of the matrix indicates the probability that it is a number 0 or 1 or 2 or .... or 10
%%		Applying the Sigmoid function (1/(1+(e^(-theta*X)))) brings the value of each element to between 0 and 1
%%
%%		For each sample - we have 10 possible values at this point - the maximum value (the one closest to 1)
%%		indicates the number that this sample actually represents
%%

z = X*all_theta';		% size (mxk) --> (5000x10) size........ 	One row per classification (1-10)
h = sigmoid(z);			% size (mxk) --> (5000x10) size

% 	for each row, find max value and its position, size of p and max_val is (5000x1) - one per row, or per classification
%   max_val gives us the computed probability, "p" or position is the number that we are predicting this is
[max_val p] = max(h, [], 2);





% =========================================================================


end
