function net = cnnff(net, x)
    n = numel(net.layers);
    net.layers{1}.a{1} = x; % x: batchsize
    inputmaps = 1;

    for l = 2 : n   %  for each layer
        if strcmp(net.layers{l}.type, 'c')                     
            %  !!below can probably be handled by insane matrix operations
            for j = 1 : net.layers{l}.outputmaps   %  for each output map
                %  create temp output map
                z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);% size after convolve
                for i = 1 : inputmaps   %  for each input map
                    %{
                    %-----------------------------dropout--------------------------------
                    if net.dropoutFraction > 0
                        if net.testing == 1
                            net.layers{l - 1}.a{i} = net.layers{l - 1}.a{i}.*(1 - net.dropoutFraction);
                        else
                            net.layers{l - 1}.dropOutMask{i} = (rand(size(net.layers{l - 1}.a{i}))>net.dropoutFraction);
                            net.layers{l - 1}.a{i} =net.layers{l - 1}.a{i}.*net.layers{l - 1}.dropOutMask{i};
                        end
                    end
                    %-------------------------------------------------------------------
                    %}
                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');%k{i,j} filter of i iput and j output
                end
                %  add bias, pass through nonlinearity
                switch net.activation
                    case 'Sigmoid'
                        net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j});
                    case 'Relu'
                        net.layers{l}.a{j} = ReLu(z + net.layers{l}.b{j});
                    otherwise
                        net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j});
                end
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;
        elseif strcmp(net.layers{l}.type, 's')
            %  downsample
            for j = 1 : inputmaps
                switch net.pooling_mode
                    case 'Mean'
                        z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   %  !! replace with variable
                        net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
                    case 'Max'
                        [net.layers{l}.a{j} net.layers{l}.id{j}]= MaxPooling(net.layers{l - 1}.a{j},[net.layers{l}.scale net.layers{l}.scale]);
                    otherwise
                        z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   %  !! replace with variable
                        net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
                end
            end
        end
    end

    %  concatenate all end layer feature maps into vector
    net.fv = [];
    for j = 1 : numel(net.layers{n}.a) % 12
        sa = size(net.layers{n}.a{j});
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
    end
    %  feedforward into output perceptrons
    net.o = net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2));
    if strcmp(net.output,'Sigmoid')
        net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));
    else
        net.o = softmax(net.o);
    end
end
