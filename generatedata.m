%% Generate data
DataNum = 1200;
games = zeros(DataNum,65,60, 'int8');
goodmoves = zeros(1,65,1, 'int8');
tic;
for i = 1:DataNum
    %% Initialize the game and draw the center stones
    u = zeros(8,8,'int8');
    u(4,4) = 1;
    u(5,5) = 1;
    u(4,5) = -1;
    u(5,4) = -1;
    move = zeros(64,1,'int8');
    move([1,3]) = sub2ind([8,8],[4 5],[4 5]);
    move([2,4]) = -sub2ind([8,8],[4 5],[5 4]);

    %% Play the game
    flag = 0;
    currentColor = 1; % start from black 
    h = 1/8;
    pass = 0; 
    lastIdx = 5;
    while pass < 2 && lastIdx <= 64 % exit with pass = 2
          %Record the board state
          games(i,:,lastIdx-4) = [0, reshape(u,1,[])];   
    %     put the stone and reverse stones captured
    %     [u,currentColor,pass] = AIrand(u,currentColor,pass,flag); 
    %     [u,currentColor,pass] = AIpositionvalue(u,currentColor,pass,flag);            
    %     [u,currentColor,pass,bestpt] = AItree2level(u,currentColor,pass,flag);   
    %     [u,currentColor,pass,bestpt] = AItreetop3(u,currentColor,pass,3,6,flag);   
    %     [u,currentColor,pass,bestpt] = AIMCTS(u,currentColor,pass,3000,40,flag);
    %     [u,currentColor,pass,bestpt] = AItree(u,currentColor,pass,3,flag);  
          [u,currentColor,pass,bestpt] = AIMC(u,currentColor,pass,30,2,flag);
          move(lastIdx) = sign(-currentColor)*bestpt;
    %     Record the move
          games(i,1,lastIdx-4) = move(lastIdx);
          lastIdx = lastIdx + 1;
    end
    %games(i,:,lastIdx-4) = [0, reshape(u,1,[])];   
    if sum(reshape(u,1,[]))> 0
        goodmoves = cat(3,goodmoves,games(i,:,games(i,1,:) > 0));
    elseif sum(reshape(u,1,[]))< 0
        goodmoves = cat(3,goodmoves, games(i,:,games(i,1,:) < 0));
    end
end

goodmoves = reshape(goodmoves,length(goodmoves(1,:,1)),length(goodmoves(1,1,:)));
goodmoves(:,1) = [];
goodmoves(1,:) = abs(goodmoves(1,:));

writematrix(goodmoves,'gamedata.csv')
toc;