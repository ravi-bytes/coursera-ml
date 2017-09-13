function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

        ypredict = pval < epsilon;  % Creates a vector of 1's and 0's
        #{
            Now we have:
            * (100x1) vector ypredict with 1's and 0's. 1==>anomaly, 0==>good
            * yval (1's and 0's) - actual cross validation data
            * ypredict (1's and 0's) - predicted data
            *
            * If yval = 1 and ypredict = 1      ==> tp:true positive
            * If yval = 1 and ypredict = 0      ==> fn:false negative
            * If yval = 0 and ypredict = 1      ==> fp:false positive
            * If yval = 0 and ypredict = 0      ==> tn:true negative
        #}

        tp = sum((yval==1) & (ypredict==1));
        fn = sum((yval==1) & (ypredict==0));
        fp = sum((yval==0) & (ypredict==1));
        tn = sum((yval==0) & (ypredict==0));

        %   Follow formula for precision, recall and F1 score 
        rec = tp/(tp+fn);
        prec = tp/(tp+fp);
        F1 = (2*prec*rec)/(prec+rec);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
