function [trainX, testX] = normalization(trainX, testX)

massimo=max(trainX);
minimo=min(trainX);
for i=1:size(trainX,2)
    trainX(1:size(trainX,1),i)=double(trainX(1:size(trainX,1),i)-minimo(i))/(massimo(i)-minimo(i));
end
for i=1:size(testX,2)
    testX(1:size(testX,1),i)=double(testX(1:size(testX,1),i)-minimo(i))/(massimo(i)-minimo(i));
end