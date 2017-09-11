function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables`
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);
count = zeros(K, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%


#{
	* Loop over every data point in data matrix X(mxn)
	* For every row "indx" is the value of the centroid (index - 1 or 2 or 3 or ... k)
	* Add the value of that row's coordinates to corresponding centroid element
		example - if "indx=2", X(i)'s values are added to centroids(2)
	* Increment count of that centroid index using "count" vector
	* Finally, find the mean of each summed up centroid value by 
		dividing against corresponding index count
#}
for i=1:m
	indx = idx(i);
	centroids(indx,:) = centroids(indx,:) + X(i,:);
	count(indx) = count(indx) + 1;
endfor

centroids = centroids./count;

% =============================================================


end

