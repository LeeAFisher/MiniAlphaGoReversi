function [validPosition, value, pass] = positionvalue(u,currentColor, depth)
%POSITIONVALUE Summary of this function goes here
%   Detailed explanation goes here
p = find(u(:) == 0);
if isempty(p) 
    pass = 1;
    value = [];
    validPosition = [];
    return
end
p = p(randperm(length(p)));
%%Here are the weights
magicWeights = [40 6 10 10 10 10 6 40;
                 6 1  3  3  3  3 1  6;
                10 3 10  8  8 10 3 10;
                10 3  8  0  0  8 3 10;
                10 3  8  0  0  8 3 10;
                10 3 10  8  8 10 3 10;
                 6 1  3  3  3  3 1  6;
                40 6 10 10 10 10 6 40];
emptyNum = length(p);
value = zeros(emptyNum, 1);
for i = 1:emptyNum
    [~,~, flipNum] = putstone(u, p(i), currentColor, 0);
    if flipNum
        value(i) = flipNum + 2*magicWeights(p(i));
    end
    if value(i) > 30 
        break;
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
        [tempPosition, tempValue, tempPass] = positionvalue(tempu, tempColor, depth-1); 
        if ~tempPass
            value(i) = value(i) - max(tempValue(:));
        end
    end
end