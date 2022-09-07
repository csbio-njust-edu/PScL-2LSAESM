function [TESTp_sftmax,predL,acc_softmax] = L1_SSAE_SM(trainX,trainY,testX,testY)
%% Classification using SAE_SM
TESTp_sftmax=[];
label_all_all=[];

tic
DISPLAY = true;
numClasses = 7;
hiddenSizeL1 = 100;    % Layer 1 Hidden Size
hiddenSizeL2 =50;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p"). 
lambda = 2e-5;         % weight decay parameter       
beta =3;              % weight of sparsity penalty term      

inputSize =size(trainX,2); 
    
    
    
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

addpath minFunc/;
options = struct;
options.Method = 'lbfgs';
options.maxIter = 200;
options.display = 'off';
[sae1OptTheta, cost] =  minFunc(@(p)sparseAutoencoderCost(p,...
    inputSize,hiddenSizeL1,lambda,sparsityParam,beta, trainX),sae1Theta,options);

if DISPLAY
  W1 = reshape(sae1OptTheta(1:hiddenSizeL1 * inputSize), hiddenSizeL1, inputSize);
%   display_network(W1');
end

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainX);

%  Randomly initialize the parameters
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);

[sae2OptTheta, cost] =  minFunc(@(p)sparseAutoencoderCost4(p,...
    hiddenSizeL1,hiddenSizeL2,lambda,sparsityParam,beta,sae1Features),sae2Theta,options);

if DISPLAY
  W11 = reshape(sae1OptTheta(1:hiddenSizeL1 * inputSize), hiddenSizeL1, inputSize);
  W12 = reshape(sae2OptTheta(1:hiddenSizeL2 * hiddenSizeL1), hiddenSizeL2, hiddenSizeL1);
end

[sae2Features] = feedForwardAutoencoder4(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);
          
                                    
%  Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);

softmaxLambda = 2e-5;
numClasses = 7;
softoptions = struct;
softoptions.maxIter =200;
softmaxModel = softmaxTrain(hiddenSizeL2,numClasses,softmaxLambda,...
                            sae2Features,trainY,softoptions);
saeSoftmaxOptTheta = softmaxModel.optTheta(:);

% Initialize the stack using the parameters learned
stack = cell(2,1);
% saelOptTheta,sae1ptTheta include sparse autoencoder's weight
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];%stackedAETheta


[stackedAEOptTheta, cost] =  minFunc(@(p)stackedAECost(p,inputSize,hiddenSizeL2,...
                         numClasses, netconfig,lambda, trainX, trainY),...
                        stackedAETheta,options);

toc
stackedTEST=struct;
stackedTEST.stackedAETheta=stackedAETheta;
stackedTEST.stackedAEOptTheta=stackedAEOptTheta;
stackedTEST.inputSize=inputSize;
stackedTEST.hiddenSizeL2=hiddenSizeL2;
stackedTEST.numClasses=numClasses;
stackedTEST.netconfig=netconfig;
% save(['G:\SAE_SM\', name1,'_stackedTEST'], 'stackedTEST') %save parameters

%%%%%%testing%%%%%%%%%%%%%%%%%%%%%%%%
[p,pred,activation44,activation55] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testX);
predL=pred';
acc_softmax=mean(testY  == pred');
TESTp_sftmax=[TESTp_sftmax; p'];
label_all_all=[ label_all_all;testY ]; 
end