function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%



#{
	Steps to find closest centroid.
	* Loop through every row of data matrix X (mxn)		-	Xi(1xn)
	* For each row, loop through each centroid option	-	ck(1xn)
		Centroid 1 is represented by centroid(1)
		.....
		Centroid K is represented by centroid(k)
	* Find the distance between the centroid and data point
			sum(Xi - ck)^2
	* Place the result in a matrix (idx_all) of size (mxK) - K being # of centroid options
	* Every row of idx_all represents the distance of the data
		represented by that row in X from corresponding centroid
	* At the end of both loops "idx_all" is created
		There is one row per data point/class (m)
		Each row has distance of that class from each centroid
			idx_all(1) --> distance from centroid "1"
			idx_all(2) --> distance from centroid "2"
			.........
			idx_all(k) --> distance from centroid "K"
	* For every row of idx_all, find the minimum value
		Place the index of that min value in "idx"
	* Thus - each value of "idx" represents the centroid closest to it
#}
m = (size(X,1));
idx_all = zeros(size(X,1), K);		% 	find the diff between a point and centroid here for all K centroids - find min later
for i=1:m
	Xi = X(i,:);					% 	ith row with n(=2) elements, each representing a feature
	for k=1:K 						%	loop through every centroid (count = K)
		ck = centroids(k,:);		% 	each centroid has n(=2) elements
		idx_all(i,k) = sum((Xi - ck).^2);			% 	(mxk) matrix - get idx for each row using min
	endfor
endfor

for i=1:m
	[minidx indx] = min(idx_all(i, :));
	idx(i) = indx;
endfor




% =============================================================

end

