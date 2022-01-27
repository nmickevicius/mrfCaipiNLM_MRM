function q = dictionaryMatchExtB1(rec, b1map, evol, lut, maxPointsPerMatch)

% evol - [nrf, natoms]
% lut  - [natoms, 3]

% get dimensions and reshape input
[nr,nc,nrf] = size(rec);
rec = reshape(rec,nr*nc,nrf);
b1map = b1map(:);

% quantize measured b1 map to values present in dictionary
b1dict = lut(:,3);
b1vals = unique(b1dict);
b1tmp = b1vals';
db1 = abs(b1map - b1tmp); % [nr*nc, nb1vals] with broadcasting
[~,idx] = min(db1,[],2);
qb1 = b1vals(idx);        % quantized b1 map

% allocate a vector of matching indices
didx = zeros(nr*nc,1);

% allocate proton density
pd = zeros(nr*nc,1);

for b = 1:length(b1vals)
    
    % get dictionary indices for 
    dictInds = find(b1dict == b1vals(b));
    bevol = evol(:,dictInds);
    blut = lut(dictInds,:);
    
    % find voxels with current value of B1
    inds = find(qb1 == b1vals(b));
    
    % determine number of matching procedures needed 
    nmatch = ceil(length(inds)/maxPointsPerMatch);
    
    for m = 1:nmatch
    
        % get signal evolution from maxPointsPerMatch voxels
        i1 = (m-1)*maxPointsPerMatch + 1;
        i2 = i1 + maxPointsPerMatch - 1;
        if i2 > length(inds)
            i2 = length(inds);
        end 
        
        % do the matching
        sig = rec(inds(i1:i2),:).';
        match = matchchunk(bevol.', sig);
        didx(inds(i1:i2)) = dictInds(match);
        %match = matchchunk(evol.', sig);
        %didx(inds(i1:i2)) = match;
        
        % calculate the proton density
        dict_evol = bevol(:,match);
        %dict_evol = evol(:,match); % X
        meas_evol = sig;           % Y
        pdtmp = zeros(size(dict_evol,2),1);
        for n = 1:size(dict_evol,2)
            pdtmp(n) = dict_evol(:,n)\meas_evol(:,n);
        end
        pd(inds(i1:i2)) = pdtmp;
        
    end
    
end

pd = reshape(pd,[nr,nc]);

% use the lookup table to get quantitative maps
q = zeros(nr, nc, size(lut,2));
for iq = 1:size(lut,2)
    qvec = lut(:,iq);
    qmap = reshape(qvec(didx),[nr,nc]);
    q(:,:,iq) = qmap;
end

% [t1, t2, pd, b1]
q = cat(3, q(:,:,1:2), abs(pd), q(:,:,3));

function matchout = matchchunk(dict,sig)

dict = dict ./ dimnorm(dict,2);
sig = sig ./ dimnorm(sig,1);

% Calculate inner product
innerproduct=dict(:,:)*sig(:,:);

% Take the maximum value and return the index
[~,matchout]=max(abs(innerproduct));

