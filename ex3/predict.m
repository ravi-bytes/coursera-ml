function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


%	X is mxn data matrix - each row is a sample/class with values x1 to xn
%	Add x0 with value "1" as Bias Unit for each element
X = [ones(size(X),1) X];

z1 = X*Theta1';					%% 	X[5000x401], Theta1[25x401] ==> a1[5000x25]
a2 = sigmoid(z1);				%% 	==> a2[5000x25]
a2 = [ones(size(a2),1) a2];		%% 	Add bias column so a2[5000x26]
z2 = a2*Theta2';				%% 	a2[5000x26], Theta2[10x26] ==> a2[5000x10]
h = sigmoid(z2);				%%  h[5000x10]
[max_val p] = max(h, [], 2);

#{
	Here's what is happening - input data is in the form a 5000 x 400 matrix.
	There are 5000 samples of handwritten digits (Each row of X representing 1 class or sample).
	Each sample is broken down into 400 data points (pixels).
	We compute 
		h(theta) = sigmoid(sum(x*theta)) for every sample, for every set of thetas
	Each set of thetas (there are 25 sets) represents a different fitment curve

	We thus take 25 sample fittings

	These are further reduced to 10 samples 

	Theta values are computed by varioius means - earlier in "oneVsAll.m" we computed theta by fmincg method

	At the end of these two steps we have a value of h(theta) (predicted values) in a 5000x10 matrix
	As usual - each row represents one piece of sample data.
	For each row or sample, each of the 10 values represents the probability of that record being a certain number represented by the place value
	
	Example:
	One random row of X has values:
		1.2803e-003,8.1516e-004,3.4057e-004,7.9902e-001,3.4119e-002,1.9964e-003,2.3576e-003,3.0966e-002,4.2380e-002,1.1041e-004
	We see that the maximum value of this 0.79902 occuring in place 4 ==> This record/sample has ~80% probability of being number 4

#}

% =========================================================================


end
