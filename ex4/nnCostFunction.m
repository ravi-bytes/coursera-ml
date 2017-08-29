function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1) X];		% 	add bias units to X - X is now size (5000x401)
yy = zeros(m,num_labels);		% 	5000x10 matrix of 0's

% 	In yy, one column will have a single one representing the place of that column
% 	example column 4 of yy will be [0;0;0;1;0;0;0;0;0;0] and so on
%  	Note - 10 actually represents zero

for i=1:m 				% 	loop through all the m (=5000) samples
	yy(i,y(i)) = 1;		
endfor
% 	Now, in matrix yy, Every row has 10 elements - 9 are zero and one element is 1
% 	This represents the place value indicated the value of that row for vector y
% 	Example: y(1010)=2 ==> yy(1010) = [0,1,0,0,0,0,0,0,0,0]

z2 = X*Theta1';			% 	[5000x401]x[401x25] --> z2 size is (5000x25)
a2 = sigmoid(z2);		% 	a2 size is (5000x25)
a2 = [ones(m,1) a2];	%	Add bias values of "1"s as the first element of every row so a2 size is now (5000x26)
z3 = a2*Theta2';		% 	[5000x26]x[26x10] --> z3 size is (5000x10)
h = sigmoid(z3);		% 	We have hTheta of size (5000x10)
#{
	Matrix "h" represents 10 possibilities (value 0 to value 9) for each of the 5000 sample data images.
	Since we have used sigmoid, each row has values between 0 and 1 only.
#}

logh = log(h);			% 	size (5000x10) or mxK
log1minush = log(1-h);	% 	size (5000x10) or mxK
% 	yy - size mxK (5000x10)
J1 = (yy.*logh) + ((1-yy).*log1minush);	
	% 	Elementwise operation: mxK matrix (5000x10)
	% 	Resulting matrix has probability for each example between 0-9 (10 values)
J = (-1/m)*J1;
J = sum(J);				% 	Sum over all 5000 samples - result is vector of size 10
J = sum(J);				% 	Sum over all classes - result is number


% -------------------------------------------------------------
% 	Regularizing cost function
% 	Remove the first column related to Bias Unit before Regularizing
T1 = Theta1(:,2:end);	% 	T1 is size (25x400)
T2 = Theta2(:,2:end);	% 	T2 is size (10x26)

reg = sum(sum((T1.^2), 2)) + sum(sum((T2.^2), 2));
reg = (lambda/(2*m))*reg;

J = J + reg;			% 	Regularized Cost Function value	

% -------------------------------------------------------------
% 	Gradient computation

% 	Vector of size equal to number of data points per sample (n=401)
% 	Bias unit included
a1 = zeros(size(X,2), 1);

% 	No need of looping 1:m since we are using Vector computations that achieves the same thing more efficiently

	% 	Step One - Compute a1,a2,a3 	%
	a1 = X;					% 	a1 size is (mx[n+1]) = 5000x401
	z2 = a1*Theta1';		% 	[5000x401]x[401x25] --> z2 size is (5000x25)
	a2 = sigmoid(z2);		% 	a2 size is (5000x25)
	a2 = [ones(m,1) a2];		%	Add bias values of "1"s as the first element of every row so a2 size is now (5000x26)
	z3 = a2*Theta2';		% 	[5000x26]x[26x10] --> z3 size is (5000x10)
	a3 = sigmoid(z3);		%   a3 size is (5000x10), K=10

	% 	Step Two - Begin Backward Propogation 	%
	delta3 = a3 - yy;		%	yy is a matrix mxk (5000x10) - each row (1x10) is a collection of zeros and ones
							% 	delta3 is vector size (mxk)=(5000x10)

	% 	Step Three - Compute delta2 	%
	delta2 = delta3*Theta2;	% 	delta3 is (mxk = 5000x10), Theta2 is (kx26 = 10x26) ==> delta2 is size (5000x26)
	delta2 = delta2(:, 2:end); % 	Remove bias column delta2 size is now 5000x25
	delta2 = delta2.*sigmoidGradient(z2); 	% 	delta2:(5000x25), z2:(5000x25) - result is size (5000X25)

	% 	Step Four 	%
	D1 = delta2'*a1;	%	[25x5000]*[5000x401] --> 25x401
	D2 = delta3'*a2;	%	[10x5000]*[5000x26] --> 10x26

	% 	Step Five - Accumulate and apply Regularization	%
	Lmatrix = lambda*ones(size(Theta1));  		% 	Matrix of lambda values of size Theta1=25x401
	Lmatrix(:,1) = 0;							% 	First column is zeros
	Theta1_grad = (D1 + (Lmatrix.*Theta1))/m; 	% 	D1(25x401), Lmatrix(25x401), Theta1(25x401)
	Lmatrix = lambda*ones(size(Theta2));  		% 	Matrix of lambda values of size Theta2=10x26
	Lmatrix(:,1) = 0;							% 	First column is zeros
	Theta2_grad = (D2 + (Lmatrix.*Theta2))/m;		% 	D2(10x26), Lmatrix(10x26), Theta2(10x26)

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
