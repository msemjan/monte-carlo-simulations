% This script shows how to plot a snapshot of a lattice. You should run
% load_lattice.m first to load the lattice.

markerSize      = 30;       % size of the dots for spins 
ulim            = 0.8;      % cutoff - specifies when the spin is up/down/or between, higher values = more strict
llim            = -ulim;    % cutoff
lim_            = 'auto';   % Limit for x and y axis
%lim_            = [0 32];   % limit for x and y axis
l                = 2;       %layer to plot

latticeSize = [sim_info(2) sim_info(2)];

% How to sum layers
if(true)
    % Averages of the chains
    ss1 = sum(s1+s4,3) / sim_info(2);
    ss2 = sum(s2+s5,3) / sim_info(2);
    ss3 = sum(s3+s6,3) / sim_info(2);
else
    ss1 = sum(s1(:,:, l), 3);
    ss2 = sum(s2(:,:, l), 3);
    ss3 = sum(s3(:,:, l), 3);

    % ss1 = sum(s4(:,:, l), 3);
    % ss2 = sum(s5(:,:, l), 3);
    % ss3 = sum(s6(:,:, l), 3);
end

% Constant sin and cos values
cs = cosd(60);
sn = sind(60);

% Create a lattice
lattice = zeros(2*latticeSize(1),2*latticeSize(2));
lattice(1:2:end,1:2:end) = ss1(:,:);
lattice(1:2:end,2:2:end) = ss2(:,:);
lattice(2:2:end,1:2:end) = ss3(:,:);
lattice(2:2:end,2:2:end) = NaN;

% X and Y positions of the dots for snapshot
X = repmat(1:(2*latticeSize(1)),2*latticeSize(2),1) + ...
    repmat(cs*(1:(2*latticeSize(2)))',1,2*latticeSize(1));
Y = repmat((1:(2*latticeSize(2)))*sn,1,2*latticeSize(1));
Y = reshape(Y, 2*latticeSize(1), 2*latticeSize(2));

% set colors
if false %true
    col_up = [0 0 0];
    col_dn = [1 1 0];
else
    col_up = [1 1 0];
    col_dn = [0 0 0];
end
col_no = [0.8 0.8 0.8];

% Draw the shapshot
f = figure;
hold on;

% Find and draw spins pointing down 
DN = find(lattice < -llim);
scatter(X(DN),Y(DN),markerSize, 'filled', ...
        'Marker',          'o', ...
        'MarkerEdgeColor', 'black', ...
        'MarkerFaceColor',  col_dn);
        

% Find and draw spins pointing up
UP = find(ulim < lattice);
scatter(X(UP),Y(UP),markerSize, 'filled', ...
        'Marker',          'o', ...
        'MarkerEdgeColor', 'black', ...
        'MarkerFaceColor', col_up);
    
% Spins between up and down
NON = find(llim <= lattice & lattice <= ulim);
scatter(X(NON),Y(NON),markerSize, 'filled', ...
        'Marker',          'o', ...
        'MarkerEdgeColor', 'black', ...
        'MarkerFaceColor', col_no);
xlim(lim_); ylim(lim_);    
hold off;
