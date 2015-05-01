function [er, bad] = cnntest(net, x, y)
    %  feedforward
    net.testing = 1;
    net = cnnff(net, x);
    [~, h] = max(net.o);
    [~, a] = max(y);
    bad = find(h ~= a);
    %{
    % virsualisation
    for i = 1 : numel(net.layers)
        for j = 1 : numel(net.layers{i}.a)
            imshow(net.layers{i,1}.a{1,j}(:,:,1)');
            figure;
        end
    end
    %}
    er = numel(bad) / size(y, 2);
end
