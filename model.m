function [T, Tout, output] = model(K, N, tsteps, T, plotOn)
%% lt 8/14/19 - model markov process in a neural network.

% ======== PARAMETERS
% K = 3; % number of states
% N = 2; % size of embedding
% tsteps = 10000; % how many to run
% T = rand(K);
% T = T./repmat(sum(T,1),K,1);
% % T = [0.2 0.8; 0.8 0.2]; % transition matrix

% ==== output:
% T, actual transition matrix
% Tout, empirical matrix
% output, sequence of outputs

%% if no T is given, then generate a random one
if isempty(T)
    T = rand(K);
    T = T./repmat(sum(T,1),K,1);
else
    assert(size(T,1)==K);
end

%% ======== initiate parameters
y = zeros(K,1);
y(1) = 1; % start in the first state
yzeros = zeros(K,1);
x = zeros(N,1);

Wfixed = rand(N, K); % fixed, since this is feedback
Wyx = rand(K, N); % plastic

xall = Wfixed * eye(K); % TODO; make sure is same activation function as below.

disp(['y = ']);
disp(y);
disp(['x = ']);
disp(x);
disp(['Wfixed = ']);
disp(Wfixed);
disp(['Wyx = ']);
disp(Wyx);


%% Solve for Wyx

% find Wyx such that W * x = log(T). Solving for W:
% --- if 0 probability then make small
Wyx = log(T)/xall;


%% Try to solve using

%% Run network
disp(' --- ');

output = nan(tsteps, 1);

for t=1:tsteps
    
    % 1) get new embedding
    x = Wfixed * y;
    
    % 2) transition to new state
    h = Wyx * x;
    h_ = h - evrnd(0, 1, [K, 1]);
    [~, idx] = max(h_);
    y = yzeros;
    y(idx)=1;
    
    %    disp(x);
    %    disp(h);
    %    disp(h_);
    %    disp(idx);
    %    disp(y)
    
    % disp(find(y));
    output(t) = find(y);
end

%% Get empirical transition matrix
Tout = zeros(K,K);
for i=1:length(output)-1
    Tout(output(i+1), output(i)) = Tout(output(i+1), output(i)) + 1;
end

% ---- normalize
Tout = Tout./repmat(sum(Tout,1),K,1);
disp('original T:');
disp(T);
disp('empirical T:');
disp(Tout);

%% Plot histograms
if plotOn==1
    % ---- 1) one row for each divergence
    figure;
    nrow = 4;
    ncol = ceil(K/4);
    for k=1:K
        subplot(nrow, ncol, k); hold on;
        title(['from ' num2str(k)]);
        ylabel('k = actual; r = empirical');
        plot(T(:,k), '-ok');
        plot(Tout(:,k), ':sr');
        ylim([0 1]);
    end
end

%%


