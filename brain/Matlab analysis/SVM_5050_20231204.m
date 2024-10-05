%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: Cross-validation and pseudo-simultaneous population
% activity using consecutive littermate interaction test
%
% Programmer : Gaeun Park
% Last updated: 10/24/2023
% Revision: 1.0
% Copyright 2023
%
% Comments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% to import and plot discrete data from a text file
close all; clc; clear;

% Behavior scheme : L1 (left) - L2 (right) - L1 (right) - L2 (left)
% Check the frame rate 
Framerate = 20;

%% load data
F1data = load ('G:\Miniscope data\20240422_IL35_synchronization\IL353\2_SocialR\Rename\IL353_socialR_F_left.mat');
F1data = F1data.Datastorage;

F2data = load('G:\Miniscope data\20240422_IL35_synchronization\IL353\2_SocialR\Rename\IL353_socialR_F_right.mat');
F2data = F2data.Datastorage;

Fdata = [F1data; F2data];
% Fdata = load ('G:\Miniscope data\20240314_IL32_synchronization\Calcium\IL333\2_SocialR\Rename\IL333_socialR_F_Right.mat');
% Fdata = Fdata.Datastorage;
Fpoch = Fdata (:,2);
[FstartPoints, FendPoints] = findConsecutiveOnesColumn(Fpoch);
Flength = FendPoints - FstartPoints + 1;
Fbehaviordata = [FstartPoints FendPoints];
FC = Fdata(:,6:end); 
clear Fdata;
clear Fpoch;
clear Flength; 
clear FendPoints;
clear FstartPoints;

N1data = load ('G:\Miniscope data\20240422_IL35_synchronization\IL353\2_SocialR\Rename\IL353_socialR_N_left.mat');
N1data = N1data.Datastorage;

N2data = load ('G:\Miniscope data\20240422_IL35_synchronization\IL353\2_SocialR\Rename\IL353_socialR_N_right.mat');
N2data = N2data.Datastorage;

Ndata = [N1data;N2data];

% Ndata = load ('G:\Miniscope data\20240213_IL30_synchronization\Calcium\IL305\2_SocialR\IL305_socialR_3.mat');
% Ndata = Ndata.Datastorage;
Npoch = Ndata (:,2);
[NstartPoints, NendPoints] = findConsecutiveOnesColumn(Npoch);
Nlength = NendPoints - NstartPoints + 1;
Nbehaviordata = [NstartPoints NendPoints];
NC = Ndata(:,6:end);
clear Ndata;
clear Npoch;
clear Nlength;
clear NendPoints;
clear NstartPoints;


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
 

 Binaverage = floor(Framerate./5);
 Binbout = floor(size(FconC,2)/Binaverage);
 FconCnewshape = zeros(size(FconC,1), Binbout, size(FconC,3));
 % Error solved when i = 1: Binaverage:size(FconC,2) --> i =
 % Binaverage:Binaverage: size(FConC,2)
% Loop through the original matrix and average every bin bout number of columns
for i = Binaverage:Binaverage:size(FconC, 2)-Binaverage+1
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
for i = Binaverage:Binaverage:size(NconC, 2)-Binaverage+1
    % Extract a block of binbout number of columns and average them
    block = NconC(:, i:i+Binaverage-1, :);
    averaged_block = mean(block, 2);
    
    % Assign the averaged block to the new matrix
    NconCnewshape(:, round(i./Binaverage), :) = averaged_block;
end

clear over over2 over3
clear FconC
clear Finteraction
clear i n
clear Ninteraction
clear NconC
clear averaged_block
clear block
clear Fbehaviordata
clear Nbehaviordata
clear FC NC
    
  



%% We randomly selected 75% of pseudo-trials for training a classifier and the remaining
%25% were used for testing decoding performance (decoder : linear SVM)
% Define parameters
k = 10; % Number of cross-validation folds
q = 5; % Number of population vectors to sample
T = 2 * q * size(FconCnewshape,3); % Total repetitions 
trainRatio = 0.5; % 2-fold cross validation

% Initialize variables to store performance scores
performanceScores = zeros(k, 1);

% Concatenate data 
wholeC = [FconCnewshape; NconCnewshape];
wholeB = [FconB; NconB];

%% To shuffle or not to shuffle (for null hypothesis)
%   random_indices = randperm(size(wholeC,1), size(wholeC,1));
%   wholeB = wholeB(random_indices);
% clear random_indices


for fold = 1:k
    % First, randomly allocate 75% as training set while 25% as test set
        random_indices = randperm(size(wholeC,1), size(wholeC,1));
        numTrainTrials = round(trainRatio * size(wholeC,1));
    
        trainDataIndices = random_indices(1:numTrainTrials);
        testDataIndices = random_indices(numTrainTrials + 1:end);
    
        trainDataC = wholeC(trainDataIndices, :, :);
        trainDataB = wholeB(trainDataIndices);
        testDataC = wholeC(testDataIndices, :, :);
        testDataB = wholeB(testDataIndices);

    % Second, make pseudo-population activity vector using training data to train SVM
        slices = [];
        for i = 1: size (trainDataC,2);
            slice = squeeze(trainDataC(:,i,:));
            slices = [slices; slice];
        end
        trainDataCnew = slices;
        clear slices
        clear slice
        
        trainDataBnew =[];
        for i = 1 : Binbout
            trainDataBnew = [trainDataBnew;trainDataB];
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

    % Train svm model using training data
     svmModel = fitcsvm(trainDataC, trainDataB, 'Standardize', true, 'KernelFunction', 'linear');

    % Make pseudo-population activity vector using test data set
        slices = [];
        for i = 1: size (testDataC,2);
            slice = squeeze(testDataC(:,i,:));
            slices = [slices; slice];
        end
        testDataCnew = slices;
        clear slices
        clear slice
     testDataBnew = [];
     for i = 1: Binbout
         testDataBnew = [testDataBnew; testDataB];
     end
    
     testDataCmatrix = zeros (T, q * size(testDataCnew,2));
        
     %Now, the data has been mixed up for 0 and 1. They first need to be
     %seperated in order to be qn long sized vector
     testDataC0 = testDataCnew(find(testDataB == 0),:);
     testDataC1 = testDataCnew(find(testDataB == 1),:);
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
       testDataB1 = ones (size(testDataC1matrix,1),1);
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
   filename = 'G:\Miniscope data\20240521_hM4Di_LSVM\LSVM_diff2\IL353_NvsL.mat'
   save(filename, '-v7.3')
