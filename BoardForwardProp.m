function [MLValues] = BoardForwardProp(u,W,b)
%BOARDFORWARDPROP Summary of this function goes here
%   Detailed explanation goes here
u = double(u);

%total = 1;
    a = cell(4,1);
    z = cell(3,1);
% layer 1 = input layer
    a{1}= reshape(u,1,[]);
    
    for i = 1:2
        z{i} = a{i}*W{i};
        z{i} = z{i} + b{i}';
        a{i+1} = max(z{i},0);
    end
    z{3} = a{2}*W{3};
%This is the softmax function on the activation layer
    s = exp(z{3});
    total = max(sum(s), eps);
    a{4} = s/total;
    MLValues = reshape(a{4},8,8);
end

