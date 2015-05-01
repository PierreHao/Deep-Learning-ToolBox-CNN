function mu = softmax(eta)
    % Softmax function
    % mu(i,c) = exp(eta(i,c))/sum_c' exp(eta(i,c'))

    % This file is from matlabtools.googlecode.com
    c = 1;

    tmp = exp(c*eta);
    denom = sum(tmp, 1);
    mu = bsxfun(@rdivide, tmp, denom);

end