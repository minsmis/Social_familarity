%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: Cross-validation and pseudo-simultaneous population
% activity using consecutive littermate interaction test
%
% Programmer : Gaeun Park
% Last updated: 12/06/2023
% Revision: 1.0
% Copyright 2023
%
% Comments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% to import and plot discrete data from a text file
close all; clc; clear;

% Check the frame rate 
Framerate = 25

%    filename = 'C:\Users\earthworm06\Desktop\File\Classifier\CCGP\IL382_F2vsN1.mat'

%% load data for training
Fdata = load ('C:\Users\earthworm06\Desktop\File\Classifier\CCGP\mCherry\Dataset\IL372_socialR_F_right.mat');
Fdata = Fdata.Datastorage;
Fpoch = Fdata (:,2);
[FstartPoints, FendPoints] = findConsecutiveOnesColumn(Fpoch);
Flength = FendPoints - FstartPoints + 1;
Fbehaviordata = [FstartPoints FendPoints];
FC = Fdata(:,6:end); 
FC = zscore(FC);
clear Fdata;
clear Fpoch;
clear Flength; 
clear FendPoints;
clear FstartPoints;

Ndata = load ('C:\Users\earthworm06\Desktop\File\Classifier\CCGP\mCherry\Dataset\IL372_socialR_N_right.mat');
Ndata = Ndata.Datastorage;
Npoch = Ndata (:,2);
[NstartPoints, NendPoints] = findConsecutiveOnesColumn(Npoch);
Nlength = NendPoints - NstartPoints + 1;
Nbehaviordata = [NstartPoints NendPoints];
NC = Ndata(:,6:end);
NC = zscore(NC);
clear Ndata;
clear Npoch;
clear Nlength;
clear NendPoints;
clear NstartPoints;

%% load data for test
F2data = load ('C:\Users\earthworm06\Desktop\File\Classifier\CCGP\mCherry\Dataset\IL372_socialR_F_left.mat');
F2data = F2data.Datastorage;
F2poch = F2data (:,2);
[F2startPoints, F2endPoints] = findConsecutiveOnesColumn(F2poch);
F2length = F2endPoints - F2startPoints + 1;
F2behaviordata = [F2startPoints F2endPoints];
FC2 = F2data(:,6:end); 
FC2 = zscore(FC2);
clear F2data;
clear F2poch;
clear F2length; 
clear F2endPoints;
clear F2startPoints;

N2data = load ('C:\Users\earthworm06\Desktop\File\Classifier\CCGP\mCherry\Dataset\IL372_socialR_N_left.mat');
N2data = N2data.Datastorage;
N2poch = N2data (:,2);2
[N2startPoints, N2endPoints] = findConsecutiveOnesColumn(N2poch);
N2length = N2endPoints - N2startPoints + 1;
N2behaviordata = [N2startPoints N2endPoints];
NC2 = N2data(:,6:end);
NC2 = zscore(NC2);
clear N2data;
clear N2poch;
clear N2length;
clear N2endPoints;
clear N2startPoints;



%% 
% For each subject and session, we divide data from each class of
% conditions (0 and 1) into training and test pseudo-trials, which each
% trials defined by a bout of interaction, with bout duration lasting from
% the beginning to end of a given interaction. Bout durations lasting
% longer than 1s were split into multiple 1s long pseudo-trials

Boutlength = floor(Framerate/2);
%Boutlength = floor(Framerate);
Fbehaviordata(:,3) = Fbehaviordata(:,2) - Fbehaviordata(:,1) + 1;

for n = size(Fbehaviordata,1):-1: 1
    if Fbehaviordata(n,3) < Boutlength;
        Fbehaviordata(n,:) = [];
    end
end

Fbehaviordata(:,4) = floor(Fbehaviordata(:,3)./Boutlength);
Finteraction = zeros(size(Fbehaviordata,1),2);

for n = 1: size(Fbehaviordata,1);
    Finteraction(n,1) = Fbehaviordata(n,1);
    Finteraction(n,2) = Fbehaviordata(n,1) + Boutlength;

end

% Additionally add(concatenate) split matrix which interaction bout is
% longer than Boutlength (>= 2 times)
over2 = [];
for n = 1: size(Fbehaviordata,1);
    if Fbehaviordata(n,4) >= 1;
        over = splitVector([Fbehaviordata(n,1) Fbehaviordata(n,2)],Boutlength);
        if over(end,2) - over(end,1) < Boutlength;
             over = over(1:end-1,:);
        else over = over;
        end
             over2 = [over2; over];
    end
    over = [];
end
    
 Finteraction = over2;
 FconC = zeros(size(Finteraction,1),Boutlength,size(FC,2));

 for n = 1:size(FconC,1);
     FconC(n,:,:) = FC(Finteraction(n,1):Finteraction(n,2),:);

 end
 FconB = zeros(size(FconC,1),1);

%% same for novel mouse
Nbehaviordata(:,3) = Nbehaviordata(:,2) - Nbehaviordata(:,1) + 1;

for n = size(Nbehaviordata,1):-1: 1
    if Nbehaviordata(n,3) < Boutlength;
        Nbehaviordata(n,:) = [];
    end
end

Nbehaviordata(:,4) = floor(Nbehaviordata(:,3)./Boutlength);
Ninteraction = zeros(size(Nbehaviordata,1),2);

for n = 1: size(Nbehaviordata,1);
    Ninteraction(n,1) = Nbehaviordata(n,1);
    Ninteraction(n,2) = Nbehaviordata(n,1) + Boutlength;

end

    over3 = [];
for n = 1: size(Nbehaviordata,1);
    if Nbehaviordata(n,4) >= 1;
        over = splitVector([Nbehaviordata(n,1) Nbehaviordata(n,2)],Boutlength);
        if over(end,2) - over(end,1) < Boutlength;
             over = over(1:end-1,:);
        else over = over;
        end
        over3 = [over3; over];
    end
    over = [];
end

 Ninteraction = unique(over3,'rows');
 NconC = zeros(size(Ninteraction,1),Boutlength,size(NC,2));

 for n = 1:size(NconC,1);
     NconC(n,:,:) = NC(Ninteraction(n,1):Ninteraction(n,2),:);

 end
 NconB = ones(size(NconC,1),1);

  % Compare the size of two interaction session and make them similar
 if size(FconC,1)> size(NconC,1);
         FconC2 = []; 
            indices = randperm(size(FconC,1), size(NconC,1));
            FconC2 = FconC(indices,:,:);
            indices =[];
            NconC2 = NconC;
 elseif size(FconC,1)< size(NconC,1);
          NconC2 = []; 
            indices = randperm(size(NconC,1), size(FconC,1));
            NconC2 = NconC(indices,:,:);
            indices =[];
            FconC2 = FconC;
 else FconC2 = FconC; NconC2 = NconC;

 end
    FconC = FconC2;
    NconC = NconC2;
    FconB = zeros(size(FconC,1),1);
    NconB = ones(size(NconC,1),1);


 %% Reshape FconC2 and NconC2 so that interaction bin can be averaged 
 

 Binaverage = floor(Framerate./10);
 Binbout = floor(size(FconC,2)/Binaverage);
 
   %% Same procedures for to make test vectors
        F2behaviordata(:,3) = F2behaviordata(:,2) - F2behaviordata(:,1) + 1;
        
        for n = size(F2behaviordata,1):-1: 1
            if F2behaviordata(n,3) < Boutlength;
                F2behaviordata(n,:) = [];
            end
        end
        
        F2behaviordata(:,4) = floor(F2behaviordata(:,3)./Boutlength);
        F2interaction = zeros(size(F2behaviordata,1),2);
        
        for n = 1: size(F2behaviordata,1);
            F2interaction(n,1) = F2behaviordata(n,1);
            F2interaction(n,2) = F2behaviordata(n,1) + Boutlength;
        
        end
        
        % Additionally add(concatenate) split matrix which interaction bout is
        % longer than Boutlength (>= 2 times)
        over2 = [];
        for n = 1: size(F2behaviordata,1);
            if F2behaviordata(n,4) >= 1;
                over = splitVector([F2behaviordata(n,1) F2behaviordata(n,2)],Boutlength);
                if over(end,2) - over(end,1) < Boutlength;
                     over = over(1:end-1,:);
                else over = over;
                end
                     over2 = [over2; over];
            end
            over = [];
        end
            
         F2interaction = over2;
         F2conC = zeros(size(F2interaction,1),Boutlength,size(FC2,2));
        
         for n = 1:size(F2conC,1);
             F2conC(n,:,:) = FC2(F2interaction(n,1):F2interaction(n,2),:);
        
         end
         F2conB = zeros(size(F2conC,1),1);
        
        % same for novel mouse
        N2behaviordata(:,3) = N2behaviordata(:,2) - N2behaviordata(:,1) + 1;
        
        for n = size(N2behaviordata,1):-1: 1
            if N2behaviordata(n,3) < Boutlength;
                N2behaviordata(n,:) = [];
            end
        end
        
        N2behaviordata(:,4) = floor(N2behaviordata(:,3)./Boutlength);
        N2interaction = zeros(size(N2behaviordata,1),2);
        
        for n = 1: size(N2behaviordata,1);
            N2interaction(n,1) = N2behaviordata(n,1);
            N2interaction(n,2) = N2behaviordata(n,1) + Boutlength;
        
        end
        
            over3 = [];
        for n = 1: size(N2behaviordata,1);
            if N2behaviordata(n,4) >= 1;
                over = splitVector([N2behaviordata(n,1) N2behaviordata(n,2)],Boutlength);
                if over(end,2) - over(end,1) < Boutlength;
                     over = over(1:end-1,:);
                else over = over;
                end
                over3 = [over3; over];
            end
            over = [];
        end
        
         N2interaction = unique(over3,'rows');
         N2conC = zeros(size(N2interaction,1),Boutlength,size(NC2,2));
        
         for n = 1:size(N2conC,1);
             N2conC(n,:,:) = NC2(N2interaction(n,1):N2interaction(n,2),:);
        
         end
         N2conB = ones(size(N2conC,1),1);
 

    



%% We randomly selected 75% of pseudo-trials for training a classifier and the remaining
%25% were used for testing decoding performance (decoder : linear SVM)
% Define parameters
k = 10; % Number of cross-validation folds
q = 5; % Number of population vectors to sample
T = 2 * q * size(N2conC,3); % Total repetitions 


% Initialize variables to store performance scores
performanceScores = zeros(k, 1);


%% To shuffle or not to shuffle
%  random_indices = randperm(size(wholeC,1), size(wholeC,1));
%  wholeB = wholeB(random_indices);


clear random_indices


for fold = 1:k
         FconCnewshape = zeros(size(FconC,1), Binbout, size(FconC,3));

        % Loop through the original matrix and average every bin bout number of columns
        for i = 1:Binaverage:size(FconC, 2)-Binaverage+1
            % Extract a block of binbout number of columns and average them
            block = FconC(:, i:i+Binaverage-1, :);
            averaged_block = mean(block, 2);
            
            % Assign the averaged block to the new matrix
            FconCnewshape(:, round(i./Binaverage), :) = averaged_block;
        end
            clear averaged_block
            clear block
        
         NconCnewshape = zeros(size(NconC,1), Binbout, size(NconC,3));
         
        % Loop through the original matrix and average every binbout number of columns
        for i = 1:Binaverage:size(NconC, 2)-Binaverage+1
            % Extract a block of binbout number of columns and average them
            block = NconC(:, i:i+Binaverage-1, :);
            averaged_block = mean(block, 2);
            
            % Assign the averaged block to the new matrix
            NconCnewshape(:, round(i./Binaverage), :) = averaged_block;
        end
        
        clear over over2 over3
        clear Finteraction
        clear i n
        clear Ninteraction
        clear averaged_block
        clear block
        clear Fbehaviordata
        clear Nbehaviordata
        clear FC NC
        
      
        
         %% Reshape FconC2 and NconC2 so that interaction bin can be averaged 
         
         F2conCnewshape = zeros(size(F2conC,1), Binbout, size(F2conC,3));
         
        % Loop through the original matrix and average every bin bout number of columns
        for i = 1:Binaverage:size(F2conC, 2)-Binaverage+1
            % Extract a block of binbout number of columns and average them
            block = F2conC(:, i:i+Binaverage-1, :);
            averaged_block = mean(block, 2);
            
            % Assign the averaged block to the new matrix
            F2conCnewshape(:, round(i./Binaverage), :) = averaged_block;
        end
            clear averaged_block
            clear block
        
         N2conCnewshape = zeros(size(N2conC,1), Binbout, size(N2conC,3));
         
        % Loop through the original matrix and average every binbout number of columns
        for i = 1:Binaverage:size(N2conC, 2)-Binaverage+1
            % Extract a block of binbout number of columns and average them
            block = N2conC(:, i:i+Binaverage-1, :);
            averaged_block = mean(block, 2);
            
            % Assign the averaged block to the new matrix
            N2conCnewshape(:, round(i./Binaverage), :) = averaged_block;
        end
        
        clear over over2 over3
        clear F2interaction
        clear i n
        clear N2interaction
        clear averaged_block
        clear block
        clear F2behaviordata
        clear N2behaviordata
        clear FC2 NC2

        % Concatenate training data 
        wholeC = [FconCnewshape; NconCnewshape];
        wholeB = [FconB; NconB];
        
        wholeC2 = [F2conCnewshape; N2conCnewshape];
        wholeB2 = [F2conB; N2conB];


    % Second, make pseudo-population activity vector using wholeC and whole B data to train SVM
        slices = [];
        for i = 1: size (wholeC,2);
            slice = squeeze(wholeC(:,i,:));
            slices = [slices; slice];
        end
        trainDataCnew = slices;
        clear slices
        clear slice
        
        trainDataBnew =[];
        for i = 1 : Binbout
            trainDataBnew = [trainDataBnew;wholeB];
        end

        trainDataCmatrix = zeros (T, q * size(trainDataCnew,2));
        
     %Now, the data has been mixed up for 0 and 1. They first need to be
     %seperated in order to be qn long sized vector
     trainDataC0 = trainDataCnew(find(trainDataBnew == 0),:);
     trainDataC1 = trainDataCnew(find(trainDataBnew == 1),:);
     trainDataC0matrix = zeros(round(T * size(trainDataC0,1)./size(trainDataCnew,1)), q * size(trainDataC0,2));
     trainDataC1matrix = zeros(round(T * size(trainDataC1,1)./size(trainDataCnew,1)), q * size(trainDataC1,2));

      for i = 1:size(trainDataC0matrix,1)
            indices = randperm(size(trainDataC0,1), q);
            selected_rows = trainDataC0(indices, :);
            trainDataC0matrix(i, :) = reshape(selected_rows.',1,[]);
            indices =[];
      end
            clear i indices selected_rows
        
      for i = 1:size(trainDataC1matrix,1)
            indices = randperm(size(trainDataC1,1), q);
            selected_rows = trainDataC1(indices, :);
            trainDataC1matrix(i, :) = reshape(selected_rows.',1,[]);
            indices =[];
      end
            clear i indices selected_rows
      
    % Concatenate data file and make same length of label matrix
       trainDataB0 = zeros(size(trainDataC0matrix,1),1);
       trainDataB1 = ones (size(trainDataC1matrix,1),1);
       trainDataB = [trainDataB0;trainDataB1];
       trainDataC = [trainDataC0matrix; trainDataC1matrix];

    % Randomly shuffle training data
    random_indices = randperm(size(trainDataC,1), size(trainDataC,1));
    trainDataB = trainDataB(random_indices);
    trainDataC = trainDataC(random_indices,:);


    % Train svm model using training data
     svmModel = fitcsvm(trainDataC, trainDataB, 'Standardize', true, 'KernelFunction', 'linear');

    % Make pseudo-population activity vector using test data set
        slices = [];
        for i = 1: size (wholeC2,2);
            slice = squeeze(wholeC2(:,i,:));
            slices = [slices; slice];
        end
        testDataCnew = slices;
        clear slices
        clear slice
     testDataBnew = [];
     for i = 1: Binbout
         testDataBnew = [testDataBnew; wholeB2];
     end
    
     testDataCmatrix = zeros (T, q * size(testDataCnew,2));
        
     %Now, the data has been mixed up for 0 and 1. They first need to be
     %seperated in order to be qn long sized vector
     testDataC0 = testDataCnew(find(testDataBnew == 0),:);
     testDataC1 = testDataCnew(find(testDataBnew == 1),:);
     testDataC0matrix = zeros(round(T * size(testDataC0,1)./size(testDataCnew,1)), q * size(testDataC0,2));
     testDataC1matrix = zeros(round(T * size(testDataC1,1)./size(testDataCnew,1)), q * size(testDataC1,2));

      for i = 1:size(testDataC0matrix,1)
            indices = randperm(size(testDataC0,1), q);
            selected_rows = testDataC0(indices, :);
            testDataC0matrix(i, :) = reshape(selected_rows.',1,[]);
            indices =[];
      end
            clear i indices selected_rows
        
      for i = 1:size(testDataC1matrix,1)
            indices = randperm(size(testDataC1,1), q);
            selected_rows = testDataC1(indices, :);
            testDataC1matrix(i, :) = reshape(selected_rows.',1,[]);
            indices =[];
      end
            clear i indices selected_rows
   
    % Concatenate data file and make same length of label matrix
       testDataB0 = zeros(size(testDataC0matrix,1),1);
       testDataB1 = ones(size(testDataC1matrix,1),1);
       testDataB = [testDataB0;testDataB1];
       testDataC = [testDataC0matrix; testDataC1matrix];

    % Predict on the test data
    predictions = predict(svmModel, testDataC);

    % Calculate accuracy
    accuracy = sum(predictions == testDataB) / numel(testDataB);
    performanceScores(fold) = accuracy;

end
    
% Calculate the mean performance score across k-fold cross-validation
meanPerformance = mean(performanceScores);

% Display the mean performance score
disp(['Mean Performance Score: ', num2str(meanPerformance)]);
% 

   save(filename, '-v7.3')

%% Plot UMAP from processed data 
