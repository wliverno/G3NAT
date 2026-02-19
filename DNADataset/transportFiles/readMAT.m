%%% Use this function to extract Fock and Overlap from Gaussian MAT file
function readMAT(strand_name)
    
    Fname = [strand_name, '_Fock'];
    Oname = [strand_name, '_Overlap'];
    
    format long

    % add current path
    apath = strcat(pwd,'/')
    hyakpath = apath;
    addpath(apath)
    file = [hyakpath, strand_name, '.txt'];
    
    fid = fopen(file);
    % the matrix want to read from output mat
    n1 = 'Label OVERLAP';
    n2 = 'Label ALPHA FOCK MATRIX';
    while ~feof(fid)
        line = fgets(fid);
        if contains(line, n1) 
            M1_size = textscan(line, '%*[^=]= %d');
            M1_size = M1_size{1}(end);
            M1 = textscan(fid, 'RArr= %f %f %f %f %f %f %f %f %f %f');
        end
        if contains(line, n2)
            M2_size = textscan(line, '%*[^=]= %d');
            M2_size = M2_size{1}(end);
            M2 = textscan(fid, 'RArr= %f %f %f %f %f %f %f %f %f %f'); 
        end
    end
    fclose(fid);
    
    % The output matrix before save to new .mat file 
    Overlap = tri2mat(M1, M1_size);
    Fock = tri2mat(M2, M2_size);
    
    F.(Fname) = Fock;
    O.(Oname) = Overlap;
    save([strand_name, '_Fock.mat'], '-struct', 'F'); 
    save([strand_name, '_Overlap.mat'], '-struct', 'O');
    
    EV = eig(Overlap^(-1)*Fock); 
    save([strand_name, '_eigen.mat'], 'EV');
    
	au_to_eV = 27.211396;
    
    %%%%%%%%%%%%%Load Matrices%%%%%%%%%%%%%%%%%%%%%%%%%%
    fock=['load -ASCII ', apath, strand_name,'_Fock.mat;'];
    overlap=['load -ASCII ', apath, strand_name,'_Overlap.mat;'];
    fock2=['load ', apath, strand_name,'_Fock.mat;'];
    overlap2=['load ', apath, strand_name,'_Overlap.mat;'];

    try
    eval(fock);  %if input matrix is ASCII
    catch
    eval(fock2); %if input matrix is regular mat
    end

    try
    eval(overlap);  %if input matrix is ASCII
    catch
    eval(overlap2); %if input matrix is regular mat
    end

    Fock_Mod = eval([strand_name,'_Fock']);
    Overlap_Mod = eval([strand_name,'_Overlap']);
    %% Calculating H0

    Fock_Mod = Fock_Mod * au_to_eV;
    H0 = (Overlap_Mod^(-0.5)) * Fock_Mod * (Overlap_Mod^(-0.5));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Fname = strcat(apath, strand_name, '.mat');
    save(Fname, 'H0', '-ASCII');

    eval(['load -ASCII ', Fname]);
    eval([strand_name,';']);
    save(Fname, strand_name);


    disp('Finished getHamiltonian!')
end

function M = tri2mat(triM, Msize)
    % determine lower or top triangular matrix 
    if Msize > 0
        n = Msize;
        IsLow = 0;
    elseif Msize < 0
        n = -Msize;
        IsLow = 1; 
    else
        fprintf(1, 'Incorrect input Matrix size'); 
    end
    
    % put triM into a vector 
    triM
    a = length(triM)
    b = length(triM{1})
    triV = ones(1, a*b);
    for i = 1:a
        for j = 1:b
            num = triM{i}(j);
            if isnan(num)
                triV(i + (j-1)*a) = 0;
            else 
                triV(i + (j-1)*a) = num;
            end
        end
    end
    
    % create the triangular matrix 
    vlength = n * (n+1) / 2
    actlen = a*b
    diff = vlength - a*b;
    if diff <= 0
        fprintf(1, 'Output Correctly: %d \n', diff);
    else
        fprintf(1, 'Incorrect input Matrix size : %d \n', diff); 
    end
    if IsLow
        M = triu(ones(n));
    else
        M = tril(ones(n));
    end
    M(M~=0) = 1:vlength;
    M = M + (M.') - diag(diag(M));
    M = reshape(triV(M),n,n);
end



















