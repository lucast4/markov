%% iterates over many model instances and saves outputs
clear all; close all;

K = 4; 
N = 3; % state size
tsteps = 50000;
T = [];
plotOn = 1;

[T, Tout, output] = model(K, N, tsteps, T, plotOn);

figure;
hold on;

subplot(2,2,1)
title('ground truth')
imagesc(T, [0 1])
colormap('cool')
colorbar('EastOutside')

subplot(2,2,2)
title('model sample')
imagesc(Tout, [0 1])
colorbar('EastOutside')

% save
savedir = 'saved_models_matlab';
dat.Tactual = T;
dat.Tout = Tout;
dat.tsteps = tsteps;

savename = [savedir '/dat_K' num2str(K) '_N' num2str(N) '_step' ...
    num2str(tsteps)];

save(savename, 'dat');


%% multiple models
clear all; close all;

nrepeats = 5; % how many iterations with each parameter set
Klist = [2 3 4 5 6 8 10];
Nlist = 1:15;

for K=Klist
    for N=Nlist
        for r=1:nrepeats
            
            tsteps = 10000;
            T = [];
            plotOn = 0;

            [T, Tout, output] = model(K, N, tsteps, T, 0);

            % save
            savedir = 'saved_models_matlab';
            dat.Tactual = T;
            dat.Tout = Tout;
            dat.tsteps = tsteps;

            savename = [savedir '/dat_K' num2str(K) '_N' num2str(N) '_step' ...
                num2str(tsteps) '_rep' num2str(r)];

            save(savename, 'dat');
        end
    end
end

%% ==== load and quantify different cases
