% expansion 28*28 to 32*32 with 0
function images = expansionData(x)
    images = zeros(size(x,1)+4,size(x,2)+4,size(x,3));
    for i = 1 : size(x,3)
        for m = 1 : size(x,1)
            for n = 1 : size(x,2)
                images(m,n,i) = x(m,n,i);
            end
        end
    end
end