categories = {'Construction','Pedestrian','SpeedLimit','Stop Signs','Traffic Light','Yield'};
rootFolder = 'Data';
imds = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');

%don't know if varSize is right or not
varSize = length(imds);

convl = convolution2dLayer(5, varSize, 'Padding', 2, 'BiasLearnRateFactor', 2);
convl.Weights = gpuArray(single(randn([5 5 3 varSize]) * 0.0001));

fc1 = fullyConnectedLayer(64, 'BiasLearnRateFactor', 2);
fc1.Weights = gpuArray(single(randn([ 64 576]) * 0.1));

fc2 = fullyConnectedLayer(4, 'BiasLearnRateFactor', 2);
fc2.Weights = gpuArray(single(randn([4 64]) * 0.1));

layers = [
    imageInputLayer([varSize varSize 3]);
    convl;
    maxPooling2dLayer(3, 'Stride', 2);
    reluLayer();
    convolution2dLayer(5, 32, 'Padding', 2, 'BiasLearnRateFactor', 2);
    reluLayer();
    averagePooling2dLayer(3, 'Stride', 2);
    convolution2dLayer(5, 64, 'Padding', 2, 'BiasLearnRateFactor', 2);
    reluLayer();
    averagePooling2dLayer(3, 'Stride', 2);
    fc1;
    reluLayer();
    fc2;
    softmaxLayer()
    classificationLayer()];

opts = trainingOptions('sdgm', ...
'InitialLearnRate', 0.001, ...
'LearnRateSchedule', 'piecewise', ...
'LearnRateDropFactor', 0.1, ...
'LearnRateDropPerioed', 8, ...
'L2Regularization', 0.004, ...
'MaxEpochs', 10, ...
'MiniBatchSize', 100, ...
'Verbose', true);

[net, info] = trainNetwork(imds, layers, opts);
