function [validPosition, value, pass] = NNpositionvalue(u,currentColor, depth, W, b)
%POSITIONVALUE Summary of this function goes here
%   Detailed explanation goes here

%This is a forward calculation which handles any amount of hidden layers

p = find(u(:) == 0);
if isempty(p) 
    pass = 1;
    value = [];
    validPosition = [];
    return
end
p = p(randperm(length(p)));

emptyNum = length(p);
value = zeros(emptyNum, 1);
NNValues = BoardForwardProp(u,W,b);
for i = 1:emptyNum
    [~,~, flipNum] = putstone(u, p(i), currentColor, 0);
    if flipNum
        value(i) = NNValues(p(i));
    end
end

if sum(value) == 0 
    pass = 1;
    value = 0;
    validPosition =0;
    return 
end

isValid = (value > 0);
validPosition = p(isValid);
value = value(isValid) + length(validPosition);

pass = 0;
if depth 
    for i = 1:length(validPosition)
        [tempu, tempColor] = putstone(u, validPosition(i), currentColor, 0); 
        [tempPosition, tempValue, tempPass] = NNpositionvalue(tempu, tempColor, depth-1, W, b); 
        if ~tempPass
            value(i) = value(i) - max(tempValue(:));
        end
    end
end