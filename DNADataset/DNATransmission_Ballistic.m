%%% This code was developed in Anantram's group at the University of Washington
%%% by Hashem Mohammad, Jianqing Qi et. al. Please send questions to anantmp@uw.edu and hashemm@uw.edu.

%12-13-2018%
%Ballistic Transmission for DNA strands

function  DNATransmission_Ballistic(run_num)
%% Initialization (Loads Parameters from subdir run_num)
format long

apath = strcat(pwd,'/');
addpath(apath)

d=['run' num2str(run_num)];
dir_path=  strcat(apath,d,'/');    %loads subdirectory
addpath(dir_path)
workdir = dir_path;

x=fopen('Parameters.txt')
dataArray=textscan(x,'%s','WhiteSpace','\r\n');
loc1=find(~cellfun(@isempty,strfind(dataArray{1,1},'Energy')));
strand=char(dataArray{1,1}(1));
Orbitals=dataArray{1,1}(3:loc1-1);
Orbitals=cellfun(@str2num,Orbitals)';

loc2=find(~cellfun(@isempty,strfind(dataArray{1,1},'Inject')));
Energy=dataArray{1,1}(loc1+1:loc2-1);
Energy=cellfun(@str2num,Energy)';
loc1=loc2;

loc2=find(~cellfun(@isempty,strfind(dataArray{1,1},'Extract')));
Lsite=dataArray{1,1}(loc1+1:loc2-1);
Lsite=cellfun(@str2num,Lsite)';
loc1=loc2;

loc2=find(~cellfun(@isempty,strfind(dataArray{1,1},'GammaL')));
Rsite=dataArray{1,1}(loc1+1:loc2-1);
Rsite=cellfun(@str2num,Rsite)';

loc1=loc2;
loc2=find(~cellfun(@isempty,strfind(dataArray{1,1},'GammaR')));
gammaL=dataArray{1,1}(loc1+1:loc2-1);
gammaL=str2double(cell2mat(gammaL))';
loc1=loc2;

loc2=find(~cellfun(@isempty,strfind(dataArray{1,1},'Probes')));
gammaR=dataArray{1,1}(loc1+1:loc2-1);
gammaR=str2double(cell2mat(gammaR));
loc1=loc2;

loc2=find(~cellfun(@isempty,strfind(dataArray{1,1},'Broadening')));
Dsites=dataArray{1,1}(loc1+1:loc2-1);
Dsites=cellfun(@str2num,Dsites)';

eta=dataArray{1,1}(loc2+1);
eta=str2double(cell2mat(eta));
bprobe=dataArray{1,1}(loc2+3);
bprobe=str2double(cell2mat(bprobe));

fclose(x);
clearvars loc1 loc2 dataArray x

%%%%%%%%%%%%%Load Matrices%%%%%%%%%%%%%%%%%%%%%%%%%%
Fm = ['load ',apath, strand, '.mat'];
eval(Fm);
H0 = eval(strand);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eta = 0;
sizeH = size(H0, 1);  % size of Hamiltonian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Norb = sum(Orbitals)
if(Norb ~= sizeH)
    return;
end
%%%%%%%%%%%Initialize gamma %%%%%%%%%%%%%%
sumSig = zeros(sizeH,sizeH);
sites = [Lsite Rsite];
gamma = [gammaL*ones(1,length(Lsite)) gammaR*ones(1,length(Rsite))];
% This subroutine gets the exact location of the atom in the Hamiltonian
% based on the number of orbitals per atom

for ii = 1 : length(sites)
    isite = sites(ii);
    TempLen1 = sum(Orbitals(1 : isite)) - Orbitals(isite);
    TempLen2 = sum(Orbitals(1 : isite));
    Len = TempLen2 - TempLen1;
    sumSig(TempLen1 + 1 : TempLen2, TempLen1 + 1 : TempLen2) = gamma(ii) * eye(Len);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%
sumSig = -1i * sumSig / 2;

%% Loop Initialization %%
NE = length(Energy);
%%%%%%%% Check available files from Checkpoint %%%%%%%%%%%
mat=dir([workdir 'Tran_*.mat']);
if length(mat)~=0
load(mat.name);
qq=find(T~=-1, 1, 'last' )+1;
else
    T=-1*ones(1,NE);
    qq=1;
end
%%%prepare the output name before entering the loop%%%
Tname=strcat(workdir,'Tran_',strand,'_gammaL_',num2str(gammaL),'_gammaR_',num2str(gammaR),'.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Entering the loop 

for nE = qq : NE
    nE
    E = Energy(nE)
    Gr = ((E + 1i * eta) * eye(sizeH) - H0 - sumSig) \ eye(sizeH);
    Ga = Gr';

%%%%%%%%%% Transmission %%%%%%%%%%%%%
%%%%%%%%%% transmission between Left and Right contact atoms %%%%%%%%%%%%%
Tmat = zeros(length(Lsite), length(Rsite));

for ii =  1: length(Lsite)
    isite = Lsite(ii);
    TempLeni1 = sum(Orbitals(1 : isite)) - sum(Orbitals(isite));
    TempLeni2 = sum(Orbitals(1 : isite));
    Leni = TempLeni2 - TempLeni1;
    Gammai = gammaL * eye(Leni);

    for jj = 1 : length(Rsite)
        jsite = Rsite(jj);
        TempLenj1 = sum(Orbitals(1 : jsite)) - sum(Orbitals(jsite));
        TempLenj2 = sum(Orbitals(1 : jsite));
        Lenj = TempLenj2 - TempLenj1;
        Gammaj = gammaR * eye(Lenj);

        Tmat(ii, jj) = real(trace(Gammai * Gr(TempLeni1 + 1 : TempLeni2, TempLenj1 + 1 : TempLenj2) * Gammaj * Ga(TempLenj1 + 1 : TempLenj2, TempLeni1 + 1 : TempLeni2)));
    end
end

T(nE) = sum(sum(Tmat));   %Ballistic Transmission is the sum of all Tij
%%%%%%% Save after every iteration %%%%
save(Tname,'Energy','T')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

disp('Finished Ballistic Transmission!')
