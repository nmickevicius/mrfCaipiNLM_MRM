function dispmrf(img, range, varargin)

logFlag = 0;
map = 'parula';
if ~isempty(varargin)
%     logFlag = 1;
    map = varargin{1};
end

if strcmpi(map,'parula')
    cmap = parula(256);
elseif strcmpi(map,'gray')
    cmap = gray(256);
elseif strcmpi(map,'CubeHelix')
    cmap = CubeHelix(256,0.5,-1.5,1.2,1.0);
elseif strcmpi(map,'turbo')
    cmap = turbo(256);
elseif strcmpi(map,'hot')
    cmap = hot(256);
end

if logFlag
    
    loginp = log10(img(:));
    loginp(isinf(loginp)) = 0;
    
    logrange = log10(range);
    
    dv = (max(logrange) - min(logrange))/size(cmap,1);
    vals = min(logrange):dv:(max(logrange)-dv);
    
    rinp = repmat(loginp, [1,numel(vals)]);
    rvals = repmat(vals,[length(loginp),1]);
    
    d = abs(rinp - rvals);
    
    [~,inds] = min(d,[],2);
    
    rgb = reshape(cmap(inds,:),[size(img),3]);
    
    mask = zeros(size(img));
    mask(abs(loginp) ~= 0) = 1;
    
    rgb = rgb .* repmat(mask,[1,1,3]);
    
    imshow(rgb,'InitialMagnification','fit');
else
    
    % allocate RGB display image
    rgb = zeros([numel(img), 3]);
    
    inp = img(:);
    
    colmask = zeros(size(inp));
    colmask(inp > min(range)) = 1;
    
    % rgb(inp >= max(range),:) = cmap(end,:);
    
    dv = (max(range) - min(range))/size(cmap,1);
    vals = min(range):dv:(max(range)-dv);
    
    
    rinp = repmat(inp, [1,numel(vals)]);
    rvals = repmat(vals,[length(inp),1]);
    
    d = abs(rinp - rvals);
    [~,inds] = min(d,[],2);
    
    rgb = reshape(cmap(inds,:),[size(img),3]);
    
    mask = zeros(size(img));
    mask(abs(inp) > 0) = 1;
    
    rgb = rgb.*repmat(mask,[1,1,3]);
    
    imshow(rgb,'InitialMagnification','fit');
    
    h = colorbar;
    h.FontSize = 24;
    col = sqrt(sum(get(gcf,'color').^2));
    if col < 0.5
        h.Color = [1,1,1];
    else
        h.Color = [0,0,0];
    end
    set(h, 'TickLabelInterpreter', 'latex');
    caxis(range);
    
    h = gca;
    colormap(h,cmap);
    
end



% set(gca,'fontsize',20);
% set(gca,'color','w');
% set(h,'color','w')

% dr = (max(range) - min(range))/4;
% rlabel = min(range):dr:max(range);
% set(h,'YTick',rlabel);
% set(gca,'fontsize',20);







