% 
% Metropolis algorithms for Ising antiferromagnet on kagome lattice
%  

name = date;
tTotal = tic;

% User-defined parameters
J               = -1;      % Coupling constant
boltzman        = 1;       % Boltzmann constant
latticeSize     = [10,10]; % Lattice size in x and y direction
temperature     = 0.1;     % Temperature at which we simulate
minField        = 0;       % Minimal value of the external magnetic field
maxField        = 5;       % Maximal value of the external magnetic field
dField          = 0.05;    % Magnetic step
numberOfSweeps  = 1*10^5;  % Number of Monte Carlo Sweeps

start           = 1+floor(0.2*numberOfSweeps); % Number of discarded MCS for thermalization
saveDir         = "/path/to/save/directory"    % Directory for saving data 

volume          = latticeSize(1)*latticeSize(2)*3;  % Volume
fieldInterval   = minField:dField:maxField;       
lenFieldInt     = length(fieldInterval);
sweepTime       = 0;

% Initialize the lattice - hot start
s1 = (rand(latticeSize)<0.5)*2-1;
s2 = (rand(latticeSize)<0.5)*2-1;
s3 = (rand(latticeSize)<0.5)*2-1;

% Calculation of exchange energy
e1 = (s2 + circshift(s2,[0 +1]) + s3 + circshift(s3,[+1 0]));
e2 = (s1 + circshift(s1,[0 -1]) + s3 + circshift(s3,[+1 -1]));
e3 = (s1 + circshift(s1,[-1 0]) + s2 + circshift(s2,[-1 +1]));

exchangeEnergy = sum(sum(s1.*e1 + s2.*e2 + s3.*e3))/2;
energy = zeros(numberOfSweeps,lenFieldInt);
magnetization = zeros(numberOfSweeps,lenFieldInt);

% calculation of initial value of magnetization
m1 = sum(sum(s1));
m2 = sum(sum(s2));
m3 = sum(sum(s3));
actualMagnetization = m1 + m2 + m3;

% Current energy
actualEnergy = -J*exchangeEnergy-minField*actualMagnetization;
magnetization(1,1) = actualMagnetization;
energy(1,1) = actualEnergy;


tTemp = tic;
counterTemp = 1;
inverseTemperature = 1/(boltzman*temperature);
for field = fieldInterval
    
    tSweep = tic;
    % Metropolis algoritmus
   for sweep = 1:numberOfSweeps
        % update 1th sublattice 
        sumNN = (s2 + circshift(s2,[0 +1]) + s3 + circshift(s3,[+1 0]));
        energyPerturbation = 2*s1.*(J*sumNN + field);
        expEP = exp(-energyPerturbation*inverseTemperature);
        transition = ((energyPerturbation<0) | (rand(latticeSize)<expEP));
        s1 = s1.*(-2*transition+1);
        
        actualEnergy = actualEnergy + sum(sum(energyPerturbation.*transition));
        m1 = sum(sum(s1));
        
        % update 2nd sublattice
        sumNN = (s1 + circshift(s1,[0 -1]) + s3 + circshift(s3,[+1 -1]));
        energyPerturbation = 2*s2.*(J*sumNN + field);
        expEP = exp(-energyPerturbation*inverseTemperature);
        transition = ((energyPerturbation<0) | (rand(latticeSize)<expEP));
        s2 = s2.*(-2*transition+1);
        
        actualEnergy = actualEnergy + sum(sum(energyPerturbation.*transition));
        m2 = sum(sum(s2));
        
        % updata 3rd sublattice
        sumNN = (s1 + circshift(s1,[-1 0]) + s2 + circshift(s2,[-1 +1]));
        energyPerturbation = 2*s3.*(J*sumNN + field);
        expEP = exp(-energyPerturbation*inverseTemperature);
        transition = ((energyPerturbation<0) | (rand(latticeSize)<expEP));
        s3 = s3.*(-2*transition+1);
        
        actualEnergy = actualEnergy + sum(sum(energyPerturbation.*transition));
        m3 = sum(sum(s3));
        
        actualMagnetization = m1 + m2 + m3;
        
        energy(sweep, counterTemp) = actualEnergy;
        magnetization(sweep, counterTemp) = actualMagnetization;
   end
    sweepTime = sweepTime + toc(tSweep);
    counterTemp = counterTemp + 1;
end
tempTime = toc(tTemp);

% calculating thermal averages
energySq = energy.^2;
magnetizationSq = magnetization.^2;
inverseTemperatureInterval = 1./(boltzman*fieldInterval);

mMagnetization = mean(abs(magnetization(start:end,:)));
mMagnetizationSq = mean(magnetizationSq(start:end,:));
mEnergy = mean(energy(start:end,:));
mEnergySq = mean(energySq(start:end,:));

susceptibility = inverseTemperature.*(mean(magnetizationSq( ...
    start:end,:))- mean(magnetization(start:end,:)).^2)/volume;
specificHeat = boltzman*(inverseTemperature.^2).*( ...
    mean(energySq(start:end,:)) - mean(energy(start:end,:)).^2)/volume;

% Saving data
totalTime = toc(tTotal);
fprintf('Time per Sweep:\t\t%f\nTime per temperature loop:\t%f\nTime total:\t\t%f\n',sweepTime/numberOfSweeps,tempTime,totalTime);
cd nameDir
name = strcat(name,'_ISING_',num2str(latticeSize(1)),'x',num2str( ... 
    latticeSize(2)),'MCS',num2str(numberOfSweeps),'T',num2str(temperature),'.mat');

save(strcat(saveDir, name));
