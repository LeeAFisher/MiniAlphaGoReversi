function [u,currentColor,pass,bestpt] = FisherAItree(u,currentColor,pass,depth,flag)
%% AITREE put a stone by tree search
%
%Slight modification that uses the Neural Network to search by tree. 
% 

%Import the weights once at the beginning. 
W = cell(3,1);
b = cell(2,1);
W{1} = table2array(readtable('weights1.csv'));
W{1}(1,:) = [];
W{2} = table2array(readtable('weights2.csv'));
W{2}(1,:) = [];
W{3} = table2array(readtable('weights3.csv'));
W{3}(1,:) = [];
b{1} = table2array(readtable('biases1.csv'));
b{1}(1,:) = [];
b{2} = table2array(readtable('biases2.csv'));
b{2}(1,:) = [];


if ~exist('flag','var')  % flag = 0 is used to count the possible flip
    flag = 1;     
end
%% Get all possible location and value
[validPosition,value,tempPass] = NNpositionvalue(u,currentColor,depth,W,b);
% showvalue(validPosition,value,currentColor);
if tempPass % no valid position, then pass
   pass = pass + 1;
   currentColor = - currentColor;
    bestpt = 0;
   return
end
%% Put the stone in the best position
[flipNum,bestpt] = max(value);
[u,currentColor] = putstone(u,validPosition(bestpt),currentColor,flag); 
bestpt = validPosition(bestpt);
pass = 0;