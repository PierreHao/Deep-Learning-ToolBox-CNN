function newLabels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

%filename = 'train-labels.idx1-ubyte';
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');
% dim(newLabels) = [10,:] with one value 1, others 0
newLabels = zeros(10,numLabels);
for i = 1 : numLabels
    newLabels(labels(i,1)+1,i) = 1;
end

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

end
