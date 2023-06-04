% 
% Layered version of Metropolis algorithm for stacked Ising antiferromagnet on
% kagome lattice with periodic boundry conditions in z-axis - N layers.
% 

pid = feature('getpid');
name = date;
tTotal = tic;

% User-defined parameters
dirLocation     = '/path/to/save/directory'; % Directory for saving data
J1              = -1;                        % Interactions within layers
J2              = 1;                         % Interactions between layers
field           = 0;                         % External magnetic field
boltzman        = 1;                         % Boltzmann constant
layers          = 24;                        % Number of layers, min 3 layers required... 
latticeSize     = [24,24,floor(layers/2)];   % Lattice size in x, y, and z direction
maxBeta         = 15;                        % Maximal inverse temperature
minBeta         = 0;                         % Minimal inverse temperature
dBeta           = 0.1;                       % Inverse temperature step
numberOfSweeps  = 7*10^5;                    % Number of sweeps
saveFreq        = 4;                         % Number of temperatures after which the data is saved

volume      = latticeSize(1)*latticeSize(2)*layers*3;
sweepTime   = 0;
saveTime    = 0;

% inverseTemperatureInterval = minBeta:dBeta:maxBeta;
a = 0.001; b = 0.8;
inverseTemperatureInterval(1) = 0;
inverseTemperatureInterval(2:length(minBeta:dBeta:maxBeta)) = ...
    a * exp(b * (minBeta:dBeta:(maxBeta-dBeta)));
lenTempInt = length(inverseTemperatureInterval);

% Log the beginning of the simulation
tlog = fopen(strcat(dirLocation,'/log.txt'),'a');
fprintf(tlog,strcat('[',datestr(datetime),'] Simulation started\n\tPID:Size = ',...
    num2str(pid),'\n\t',num2str(latticeSize(1)),'x',num2str(latticeSize(2)),'x',...
    num2str(layers),'\n\tTemp = ', num2str(minBeta), ':',num2str(dBeta),':',...
    num2str(maxBeta), '\n\tJ2 = ', num2str(J2),'\n'));
fclose(tlog);

% Initialize the lattice
s1 = (rand(latticeSize)<0.5)*2-1;
s2 = (rand(latticeSize)<0.5)*2-1;
s3 = (rand(latticeSize)<0.5)*2-1;
s4 = (rand(latticeSize)<0.5)*2-1;
s5 = (rand(latticeSize)<0.5)*2-1;
s6 = (rand(latticeSize)<0.5)*2-1;

% Calculation of exchange energy within layer
e1 = (s2 + circshift(s2,[0 +1 0]) + s3 + circshift(s3,[+1 0 0]));
e2 = (s1 + circshift(s1,[0 -1 0]) + s3 + circshift(s3,[+1 -1 0]));
e3 = (s1 + circshift(s1,[-1 0 0]) + s2 + circshift(s2,[-1 +1 0]));
exchangeEnergy1 = sum(sum(sum(s1.*e1 + s2.*e2 + s3.*e3)))/2;
e1 = (s5 + circshift(s5,[0 +1 0]) + s6 + circshift(s6,[+1 0 0]));
e2 = (s4 + circshift(s4,[0 -1 0]) + s6 + circshift(s6,[+1 -1 0]));
e3 = (s4 + circshift(s4,[-1 0 0]) + s5 + circshift(s5,[-1 +1 0]));
exchangeEnergy1 = exchangeEnergy1 + sum(sum(sum(s4.*e1 + s5.*e2 + s6.*e3)))/2;

% Calculation of exchange energy between layers
e1 = s4 + circshift(s4,[0 0 1]);
e2 = s5 + circshift(s5,[0 0 1]);
e3 = s6 + circshift(s6,[0 0 1]);
exchangeEnergy2 = sum(sum(sum(s1.*e1 + s2.*e2 + s3.*e3)))/2;
e1 = s1 + circshift(s1,[0 0 -1]);
e2 = s2 + circshift(s2,[0 0 -1]);
e3 = s3 + circshift(s3,[0 0 -1]);
exchangeEnergy2 = exchangeEnergy2 + sum(sum(sum(s4.*e1 + s5.*e2 + s6.*e3)))/2;

energy = zeros(numberOfSweeps,lenTempInt);
m1 = zeros(numberOfSweeps,lenTempInt);
m2 = zeros(numberOfSweeps,lenTempInt);
m3 = zeros(numberOfSweeps,lenTempInt);
m4 = zeros(numberOfSweeps,lenTempInt);
m5 = zeros(numberOfSweeps,lenTempInt);
m6 = zeros(numberOfSweeps,lenTempInt);

% calculation of initial value of magnetization
m1(1,1) = sum(sum(sum(s1)));
m2(1,1) = sum(sum(sum(s2)));
m3(1,1) = sum(sum(sum(s3)));
m4(1,1) = sum(sum(sum(s4)));
m5(1,1) = sum(sum(sum(s5)));
m6(1,1) = sum(sum(sum(s6)));
actualMagnetization = m1(1,1) + m2(1,1) + m3(1,1) + m4(1,1) + m5(1,1) + ...
    m6(1,1);
actualEnergy = -J1*exchangeEnergy1-J2*exchangeEnergy2-field*actualMagnetization;
energy(1,1) = actualEnergy;

clear e1 e2 e3 a b;

sumNN1 = zeros(latticeSize);
sumNN2 = zeros(latticeSize);

name = strcat(dirLocation,'/',name,'_ISING_',num2str(latticeSize(1)),'x',num2str( ... 
    latticeSize(2)),'x',num2str(layers),'MCS',num2str(numberOfSweeps),'F',...
    num2str(field),'J2',num2str(J2),'.mat');

try

    tTemp = tic;
    counterTemp = 1;
    for inverseTemperature = inverseTemperatureInterval
        tSweep = tic;
        % Metropolis algoritmus
       for sweep = 1:numberOfSweeps
            % update 1th sublattice 
            sumNN1 = (s2 + circshift(s2,[0 +1 0]) + s3 + circshift(s3,[+1 0 0]));
            sumNN2 = s4 + circshift(s4,[0 0 1]);
            energyPerturbation = 2*s1.*(J1*sumNN1 + J2*sumNN2 + field);
            expEP = exp(-energyPerturbation*inverseTemperature);
            transition = ((energyPerturbation<0) | (rand(latticeSize)<expEP));
            s1 = s1.*(-2*transition+1);

            actualEnergy = actualEnergy + sum(sum(sum(energyPerturbation.*transition)));
            m1(sweep, counterTemp) = sum(sum(sum(s1)));

            % update 2nd sublattice
            sumNN1 = (s1 + circshift(s1,[0 -1 0]) + s3 + circshift(s3,[+1 -1 0]));
            sumNN2 = s5 + circshift(s5,[0 0 1]);
            energyPerturbation = 2*s2.*(J1*sumNN1 + J2*sumNN2 + field);
            expEP = exp(-energyPerturbation*inverseTemperature);
            transition = ((energyPerturbation<0) | (rand(latticeSize)<expEP));
            s2 = s2.*(-2*transition+1);

            actualEnergy = actualEnergy + sum(sum(sum(energyPerturbation.*transition)));
            m2(sweep, counterTemp) = sum(sum(sum(s2)));

            % updata 3rd sublattice
            sumNN1 = (s1 + circshift(s1,[-1 0 0]) + s2 + circshift(s2,[-1 +1 0]));
            sumNN2 = s6 + circshift(s6,[0 0 1]);
            energyPerturbation = 2*s3.*(J1*sumNN1 + J2*sumNN2 + field);
            expEP = exp(-energyPerturbation*inverseTemperature);
            transition = ((energyPerturbation<0) | (rand(latticeSize)<expEP));
            s3 = s3.*(-2*transition+1);

            actualEnergy = actualEnergy + sum(sum(sum(energyPerturbation.*transition)));
            m3(sweep, counterTemp) = sum(sum(sum(s3)));
            
            % update 4th sublattice
            sumNN1 = (s5 + circshift(s5,[0 +1 0]) + s6 + circshift(s6,[+1 0 0]));
            sumNN2 = s1 + circshift(s1,[0 0 -1]);
            energyPerturbation = 2*s4.*(J1*sumNN1 + J2*sumNN2 + field);
            expEP = exp(-energyPerturbation*inverseTemperature);
            transition = ((energyPerturbation<0) | (rand(latticeSize)<expEP));
            s4 = s4.*(-2*transition+1);
            
            actualEnergy = actualEnergy + sum(sum(sum(energyPerturbation.*transition)));
            m4(sweep, counterTemp) = sum(sum(sum(s4)));
            
            % update 5th sublattice
            sumNN1 = (s4 + circshift(s4,[0 -1 0]) + s6 + circshift(s6,[+1 -1 0]));
            sumNN2 = s2 + circshift(s2,[0 0 -1]);
            energyPerturbation = 2*s5.*(J1*sumNN1 + J2*sumNN2 + field);
            expEP = exp(-energyPerturbation*inverseTemperature);
            transition = ((energyPerturbation<0) | (rand(latticeSize)<expEP));
            s5 = s5.*(-2*transition+1);
            
            actualEnergy = actualEnergy + sum(sum(sum(energyPerturbation.*transition)));
            m5(sweep, counterTemp) = sum(sum(sum(s5)));
            
            % update 6th sublattice
            sumNN1 = (s4 + circshift(s4,[-1 0 0]) + s5 + circshift(s5,[-1 +1 0]));
            sumNN2 = s3 + circshift(s3,[0 0 -1]);
            energyPerturbation = 2*s6.*(J1*sumNN1 + J2*sumNN2 + field);
            expEP = exp(-energyPerturbation*inverseTemperature);
            transition = ((energyPerturbation<0) | (rand(latticeSize)<expEP));
            s6 = s6.*(-2*transition+1);
            
            actualEnergy = actualEnergy + sum(sum(sum(energyPerturbation.*transition)));
            m6(sweep, counterTemp) = sum(sum(sum(s6)));
            
            energy(sweep, counterTemp) = actualEnergy;

       end
        sweepTime = sweepTime + toc(tSweep);
        counterTemp = counterTemp + 1;
        
        % Log the end of sweep loop 
        tlog = fopen(strcat(dirLocation,'/log.txt'),'a');
        fprintf(tlog,'\t%d/%d done...\n',counterTemp,lenTempInt);
        fclose(tlog);
       
        % Save data
        tSave = tic;
        if(rem(counterTemp, saveFreq)==0)
            save(name,'s1','s2','s3','s4','s5','s6',...
                'm1','m2','m3','m4','m5','m6','inverseTemperature',...
                'inverseTemperatureInterval','counterTemp','energy',...
                'pid');
        end
        saveTime = saveTime + toc(tSave);
    end
    tempTime = toc(tTemp);

    % Calculate thermal averages
    energySq = energy.^2;
    magnetization = m1 + m2 + m3 +m4 + m5 + m6;
    magnetizationSq = magnetization.^2;

    start = 1+floor(0.2*numberOfSweeps);
    mm1 = mean(abs(m1(start:end,:)));
    mm2 = mean(abs(m2(start:end,:)));
    mm3 = mean(abs(m3(start:end,:)));
    mm4 = mean(abs(m4(start:end,:)));
    mm5 = mean(abs(m5(start:end,:)));
    mm6 = mean(abs(m6(start:end,:)));
    mMagnetization = mm1 + mm2 + mm3 + mm4 +mm5 + mm6;
    mMagnetizationSq = mean(magnetizationSq(start:end,:));
    mEnergy = mean(energy(start:end,:));
    mEnergySq = mean(energySq(start:end,:));

    entropy = volume*log(2) + inverseTemperatureInterval.*mEnergy - ...
        cumtrapz(inverseTemperatureInterval,mEnergy);

    susceptibility = inverseTemperatureInterval.*(mean(magnetizationSq( ...
        start:end,:))- mean(magnetization(start:end,:)).^2)/volume;
    specificHeat = boltzman*(inverseTemperatureInterval.^2).*( ...
        mean(energySq(start:end,:)) - mean(energy(start:end,:)).^2)/volume;

    % Saving data
    totalTime = toc(tTotal);
    
    tSave = tic;
    save(name,'s1','s2','s3','s4','s5','s6',...
        'm1','m2','m3','m4','m5','m6','inverseTemperatureInterval','energy',...
        'magnetization','mm1','mm2','mm3','mm4','mm5','mm6','mMagnetization',...
        'mEnergy','entropy','susceptibility','specificHeat','pid','latticeSize',...
        'volume','field','J1','J2');

    saveTime = saveTime + toc(tSave);
    
    tlog = fopen(strcat(dirLocation,'/log.txt'),'a');
    logStr = strcat('[',datestr(datetime),'] Simulation ended\n\tTime per sweep:\t', ...
        num2str(sweepTime/numberOfSweeps),' s/sweep\n\tTime per temperature:\t',...
        num2str(tempTime),' s/temperature\n\tTime total:\t',num2str(totalTime),...
        ' s\n\tTime to save data:\t',num2str(saveTime),'\n');
    fprintf(tlog, logStr);
    fclose(tlog);

catch ME
    tlog = fopen(strcat(dirLocation,'/log.txt'),'a');
    logStr = strcat('[',datestr(datetime),'] Error: ',num2str(ME.stack.line),...
        ':\t',ME.message,'\n');
    fprintf(tlog, logStr);
    fclose(tlog);
    rethrow(ME);
end    
