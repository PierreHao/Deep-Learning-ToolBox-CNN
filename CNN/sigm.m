function X = sigm(P)
    X = 1./(1+exp(-P));
    %X = max(0,P);
end