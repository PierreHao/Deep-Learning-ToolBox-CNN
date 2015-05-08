%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tutor : PierreHao (Hao ZHANG)
% Company : NanJing Qingsou 
% Note : if you publish this code, do not forget @me 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
%{
clc;
clear all
load mnist_uint8;

train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');
%}
filenameTrain = 'train-images.idx3-ubyte';
filenameTrainLabel = 'train-labels.idx1-ubyte';
filenameTest = 't10k-images.idx3-ubyte';
filenameTestLabel = 't10k-labels.idx1-ubyte';

% load MNIST 
train_x = loadMNISTImages(filenameTrain);
test_x = loadMNISTImages(filenameTest);
train_y = loadMNISTLabels(filenameTrainLabel);
test_y = loadMNISTLabels(filenameTestLabel);

% normalization 0.47
train_x = double(train_x)/255;
test_x = double(test_x)/255;
train_y = double(train_y);
test_y = double(test_y);
% expand to 32*32 
train_x = expansionData(train_x);
test_x = expansionData(test_x);

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11.30% error. 
%With 100 epochs you'll get around 1.2% error

rand('state',0)

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer outputmaps: numbers of filters
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer 
    struct('type', 's', 'scale', 2) %subsampling layer
    %struct('type', 'c', 'outputmaps', 120, 'kernelsize', 5) %0.0388
    %struct('type', 's', 'scale', 2) %subsampling layer
};


opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 1;
cnn.testing = 0;
cnn.activation = 'Sigmoid'; % now we have Relu and Sigmoid activation functions
cnn.pooling_mode = 'Mean'; %now we have Mean and Max pooling
cnn.output = 'Softmax';% noe we have Softmax and Sigmoid output function
opts.iteration = 1;
cnn = cnnsetup(cnn, train_x, train_y);
for i = 1 : opts.iteration
    cnn = cnntrain(cnn, train_x, train_y, opts);
    [er, bad] = cnntest(cnn, test_x, test_y);
    fprintf('%d iterations and rate of error : %d\n',i,er);
    %[er1, bad1] = cnntest(cnn, val_x, val_Label);
    %fprintf('%d iterations and rate of error (validation) : %d\n',i,er1);
    %if mod(i,10) == 0
    %if mod(i,3) == 0
    %    opts.alpha = opts.alpha/10;%change learning rate
    %end
end
%[er, bad] = cnntest(cnn, test_x, test_y);
fprintf('Taux of error : %d\n',er(i));
%plot mean squared error
figure; plot(cnn.rL);
assert(er<0.12, 'Too big error');
