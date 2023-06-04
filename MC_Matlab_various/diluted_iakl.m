% 
%  This code implements Metropolis Algorithm for Ising model 
%  on Kagome lattice with selective dilution on the third sublattice.
% 

% Declaring where to save our data
dirLocation = '/path/to/save/directory';

% PID, just to be sure. You never know where you need to kill your process 
% in cold-blood...
pid = feature('getpid');
fprintf('my pid is: %d\n',pid);
name = date;

% Declaration of the simulation parameters
tTotal = tic;
J           = -1;           % Coupling constant
boltzman    = 1;            % Boltzmann factor
field       = 0;            % External magnetic field
dilution    = 0.1;          % (Quenched) Dilution on the third sublattice 
latticeSize = [32,32];      % Lattice size in x and y direction
numberOfSweeps = 1*10^6;    % Number of sweeps
saveFreq    = 25;           % Saving frequency - 25 means that the data will be
                            % saved after every 25-th temperature cycle
volume      = latticeSize(1)*latticeSize(2)*3;

sweepTime   = 0;

% Logging the beginning of the simulation
tlog = fopen(strcat(dirLocation,'/log.txt'),'a');
fprintf(tlog,strcat('[',datestr(datetime),'] Simulation of Ising model', ...
    ' on diluted Kagome lattice started\n\tPID:Size = ',...
    num2str(pid),'\n\t',num2str(latticeSize(1)),'x', ... 
    num2str(latticeSize(2)), '\n\tTemp = ', num2str(minBeta), ':', ...
    num2str(dBeta),':', num2str(maxBeta), '\n\tJ = ', num2str(J), ...
    '\n\ta = ', num2str(a),'\tb = ',num2str(b), '\n\tDilution = ', ...
    num2str(dilution), '\n'));
fclose(tlog);

% Setting the initial state of the lattice
s1 = (rand(latticeSize)<0.5)*2-1;
s2 = (rand(latticeSize)<0.5)*2-1;
s3 = (rand(latticeSize)<0.5)*2-1;

% Dilution percent of spins are going to be non-magnetic (achieved by setting them to 0)
s3(rand(latticeSize)<dilution) = 0; 

% The Calculation of the initial exchange energy
e1 = (s2 + circshift(s2,[0 +1]) + s3 + circshift(s3,[+1 0]));
e2 = (s1 + circshift(s1,[0 -1]) + s3 + circshift(s3,[+1 -1]));
e3 = (s1 + circshift(s1,[-1 0]) + s2 + circshift(s2,[-1 +1]));
exchangeEnergy = sum(sum(s1.*e1 + s2.*e2 + s3.*e3))/2;
energy = zeros(numberOfSweeps,lenTempInt);

% The Calculation of the initial value of magnetization
magnetization = zeros(numberOfSweeps,lenTempInt);
m1 = zeros(numberOfSweeps,lenTempInt);
m2 = zeros(numberOfSweeps,lenTempInt);
m3 = zeros(numberOfSweeps,lenTempInt);
m1(1,1) = sum(sum(s1));
m2(1,1) = sum(sum(s2));
m3(1,1) = sum(sum(s3));
actualMagnetization = m1(1,1) + m2(1,1) + m3(1,1);

% The current energy
actualEnergy = -J*exchangeEnergy-field*actualMagnetization;
magnetization(1,1) = actualMagnetization;
energy(1,1) = actualEnergy;

% Ordering the sublattice magnetizations 
minM = zeros(numberOfSweeps,lenTempInt);
midM = zeros(numberOfSweeps,lenTempInt);
maxM = zeros(numberOfSweeps,lenTempInt);

minM(1,1) = min([m1(1,1), m2(1,1), m3(1,1)]);
midM(1,1) = median([m1(1,1), m2(1,1), m3(1,1)]);
maxM(1,1) = max([m1(1,1), m2(1,1), m3(1,1)]);     

% Order of plaquettes tells us wether each triangle plaquette is in state 
% where two spins are antiparallel and the third one is random (than it is 1)
% or if it is in paramagnetic configuration (parameter is 3/2)
orderOfPlaquettes      = zeros(numberOfSweeps,lenTempInt);
orderOfPlaquettes(1,1) = (sum(sum(sum(abs(s1+s2+s3)))) + ...
                          sum(sum(sum(abs(circshift(s1,[-1 +1 0]) + ...
                                      circshift(s2,[-1 0 0])+s3)))))/...
                         (latticeSize(1)*latticeSize(2)*2);

% We assume that when dilution is equal to 1 (all spins in the third 
% sublattice are non-magnetic), we will have a non-interacting set of 
% antiferromagnetic linear chains - this parameter is basically staggered
% magnetization of the remaining two sublattices
chainOrder = zeros(numberOfSweeps, lenTempInt);
chainOrder(1,1) = sum(abs(sum(s1-s2,2)));
clear e1 e2 a b;

% Auxiliary variable
sumNN = zeros(latticeSize);

% Setting up name/path to our data...
name = strcat(dirLocation,'/',name,'_DILUTED_KAGOME_', ... 
              num2str(latticeSize(1)),'x',num2str(latticeSize(2)),...
              'MCS',num2str(numberOfSweeps),'F',num2str(field), ...
              'J',num2str(J),...
              'p',num2str(dilution),...
              'PID', num2str(pid), ...
              'num', num2str(num), ...
              '.mat');

tTemp = tic;
counterTemp = 1;
saveTime = 0;

try
    % Real magic happens here... Simulation is starting.
    for inverseTemperature = inverseTemperatureInterval
        
       tSweep = tic;
       % Metropolis algoritmus
       for sweep = 1:numberOfSweeps
            % update 1th sublattice 
            sumNN = (s2 + circshift(s2,[0 +1]) + s3 + circshift(s3,[+1 0]));
            energyPerturbation = 2*s1.*(J*sumNN + field);
            expEP = exp(-energyPerturbation*inverseTemperature);
            transition = ((energyPerturbation<0) | ...
                          (rand(latticeSize)<expEP));
            s1 = s1.*(-2*transition+1);
            
            actualEnergy = actualEnergy + sum(sum(energyPerturbation.* ...
                                                  transition));
            m1(sweep, counterTemp) = sum(sum(s1));
            
            % update 2nd sublattice
            sumNN = (s1 + circshift(s1,[0 -1]) + s3 + circshift(s3,[+1 -1]));
            energyPerturbation = 2*s2.*(J*sumNN + field);
            expEP = exp(-energyPerturbation*inverseTemperature);
            transition = ((energyPerturbation<0) | ...
                          (rand(latticeSize)<expEP));
            s2 = s2.*(-2*transition+1);
            
            actualEnergy = actualEnergy + sum(sum(energyPerturbation.* ...
                                                  transition));
            m2(sweep, counterTemp) = sum(sum(s2));
            
            % updata 3rd sublattice
            sumNN = (s1 + circshift(s1,[-1 0]) + s2 + circshift(s2,[-1 +1]));
            energyPerturbation = 2*s3.*(J*sumNN + field);
            expEP = exp(-energyPerturbation*inverseTemperature);
            transition = ((energyPerturbation<0) | ...
                          (rand(latticeSize)<expEP));
            s3 = s3.*(-2*transition+1);
            
            actualEnergy = actualEnergy + sum(sum(energyPerturbation.* ...
                                                  transition));
            m3(sweep, counterTemp) = sum(sum(s3));
            
            % Ordering sublattice magnetizations
            minM(sweep, counterTemp) = min([m1(sweep, counterTemp), ...
                                            m2(sweep, counterTemp), ...
                                            m3(sweep, counterTemp)]);
            midM(sweep, counterTemp) = median([m1(sweep, counterTemp), ...
                                            m2(sweep, counterTemp), ...
                                            m3(sweep, counterTemp)]);
            maxM(sweep, counterTemp) = max([m1(sweep, counterTemp), ...
                                            m2(sweep, counterTemp), ...
                                            m3(sweep, counterTemp)]);                                        
                                        
                                        
            % Calculation of several quantities
            magnetization(sweep, counterTemp) = m1(sweep, counterTemp) + ...
                                                m2(sweep, counterTemp) + ...
                                                m3(sweep, counterTemp);       
            
            orderOfPlaquettes(sweep, counterTemp) = ...
                             (sum(sum(sum(abs(s1+s2+s3)))) + ...
                              sum(sum(sum(abs(circshift(s1,[-1 +1 0]) + ...
                                          circshift(s2,[-1 0 0])+s3)))))/...
                             (latticeSize(1)*latticeSize(2)*2);
                
            chainOrder(sweep, counterTemp) = sum(abs(sum(s1-s2,2)));
            energy(sweep, counterTemp) = actualEnergy;
        end % End of the sweep cycle
        
        % Logging
        tlog = fopen(strcat(dirLocation,'/log.txt'),'a');
        fprintf(tlog,'\t[%s] PID%d: %d/%d done...\n',datestr(datetime), ...
            pid,counterTemp,lenTempInt);
        fclose(tlog);

        % Saving data each saveFreq temperature - to speed up calculation
        if(rem(counterTemp, saveFreq)==0)
          tSave = tic;
            save(name,'s1','s2','s3','chainOrder', ...
                'm1','m2','m3','inverseTemperature',...
                'inverseTemperatureInterval','counterTemp','energy',...
                'pid','orderOfPlaquettes');

            saveTime = saveTime + toc(tSave);
        end

        sweepTime = sweepTime + toc(tSweep);
        counterTemp = counterTemp + 1;
    end % End of the temperature cycle
    tempTime = toc(tTemp);

    % Calculation of squares of some quantities
    energySq = energy.^2;
    magnetizationSq = magnetization.^2;

    % Calculation of mean values
    start = 1+floor(0.2*numberOfSweeps);
    mm1 = mean(abs(m1(start:end,:)));
    mm2 = mean(abs(m2(start:end,:)));
    mm3 = mean(abs(m3(start:end,:)));
    mMagnetization = mean(abs(magnetization(start:end,:)));
    mMagnetizationSq = mean(magnetizationSq(start:end,:));
    mEnergy = mean(energy(start:end,:));
    mEnergySq = mean(energySq(start:end,:));
    
    % Calculation of entropy using thermodynamic integration method
    entropy = volume*log(2) + inverseTemperatureInterval.*mEnergy - ...
        cumtrapz(inverseTemperatureInterval,mEnergy);

    % Calculation of more thermodynamic quantities
    susceptibility = inverseTemperatureInterval.*(mean(magnetizationSq( ...
        start:end,:))- mean(magnetization(start:end,:)).^2)/volume;
    specificHeat = boltzman*(inverseTemperatureInterval.^2).*( ...
        mean(energySq(start:end,:)) - mean(energy(start:end,:)).^2)/volume;

    % Saving our amazing data...
    tSave = tic;
    save(name);
    saveTime = saveTime + toc(tSave);

    % Outputting the duration of the simulation
    totalTime = toc(tTotal);
    fprintf(strcat('Time per Sweep:\t\t', num2str(sweepTime/ ...
        numberOfSweeps), ...
        '\nTime per temperature loop:\t', num2str(tempTime), ...
        '\nTime total:\t\t',num2str(totalTime),'\n'));

    % Logging the duration of the simulation
    tlog = fopen(strcat(dirLocation,'/log.txt'),'a');
    logStr = strcat('[',datestr(datetime),'] Simulation ended\n\tTime ',...
    'per sweep:\t',num2str(sweepTime/numberOfSweeps),' s/sweep\n\tTime ',...
    'per temperature:\t',num2str(tempTime),' s/temperature\n\tTime ',...
    'total:\t',num2str(totalTime),' s\n\tTime to save data:\t',num2str(...
    saveTime),'\n');
    fprintf(tlog, logStr);
    fclose(tlog);

catch ME
    tlog = fopen(strcat(dirLocation,'/log.txt'),'a');
    logStr = strcat('[',datestr(datetime),'] Error\n\tPID',num2str(pid),...
    '\t',num2str(ME.stack.line),':\t',ME.message,'\n');
    fprintf(tlog, logStr);
    fclose(tlog);
    rethrow(ME);
end    
