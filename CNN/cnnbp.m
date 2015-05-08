function net = cnnbp(net, y)
    n = numel(net.layers);

    %   error
    net.e = net.o - y;
    %  loss function
    net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2); % mean of batchSize images
    %net.L = -sum(sum(y .* log(net.o))) / size(net.e, 2);
    %  backprop deltas
    if strcmp(net.output,'Sigmoid')
        net.od = net.e .* (net.o .* (1 - net.o));
    else
        net.od = net.e;
    end
    %if we use ReLU, the last layer is not ReLU
    net.fvd = (net.ffW' * net.od);              %  feature vector delta
    if strcmp(net.layers{n}.type, 'c')         %  only conv layers has sigm function
        switch net.activation
            case 'Sigmoid'
                net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
            case 'Relu'
                net.fvd = net.fvd.*(net.fv > 0);
            otherwise
                net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
        end
    end

    %  reshape feature vector deltas into output map style
    sa = size(net.layers{n}.a{1});
    fvnum = sa(1) * sa(2);
    for j = 1 : numel(net.layers{n}.a)
        net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    end

    for l = (n - 1) : -1 : 1
        if strcmp(net.layers{l+1}.type, 's')%&& strcmp(net.layers{l}.type, 'c') % pay attention, we can add strcmp(net.layers{l}.type, 'c') 
            for j = 1 : numel(net.layers{l}.a)
                if strcmp(net.pooling_mode,'Mean')
                    switch net.activation
                        case 'Sigmoid'
                            net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) .* (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) / net.layers{l + 1}.scale ^ 2);
                        case 'Relu'
                            net.layers{l}.d{j} =(expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) / (net.layers{l + 1}.scale ^ 2)).*(net.layers{l}.a{j}>0);
                        otherwise
                            net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) .* (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) / net.layers{l + 1}.scale ^ 2);
                    end
                
                elseif strcmp(net.pooling_mode,'Max')
                    switch net.activation
                        case 'Sigmoid'
                            up = zeros(size(net.layers{l}.a{1}));
                            up(net.layers{l+1}.id{j}) = net.layers{l+1}.d{j}(:);% up is matrix of delta:d not a
                            net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) .* up;
                        case 'Relu'
                            up = zeros(size(net.layers{l}.a{1}));
                            up(net.layers{l+1}.id{j}) = net.layers{l+1}.d{j}(:);
                            net.layers{l}.d{j} =up.*(net.layers{l}.a{j}>0);
                        otherwise
                            up = zeros(size(net.layers{l}.a{1}));
                            up(net.layers{l+1}.id{j}) = net.layers{l+1}.d{j}(:);
                            net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) .* up;
                    end
                end 
            end
        elseif strcmp(net.layers{l+1}.type, 'c')%&& strcmp(net.layers{l}.type, 's')
            for i = 1 : numel(net.layers{l}.a)
                z = zeros(size(net.layers{l}.a{1}));
                for j = 1 : numel(net.layers{l + 1}.a)
                     z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                end
                net.layers{l}.d{i} = z;
                %{
                %--------------------------dropout-------------------------------
                if net.dropoutFraction>0
                    net.layers{l}.d{i} = net.layers{l}.d{i} .* net.layers{l}.dropOutMask{i};
                end
                %-----------------------------------------------------------------
                %}
            end
        end
    end

    %%  calc gradients
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for i = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
        end
    end
    net.dffW = net.od * (net.fv)' / size(net.od, 2);
    net.dffb = mean(net.od, 2);

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end
