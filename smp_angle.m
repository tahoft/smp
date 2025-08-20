%% smp_angle
% Extract angle of 'U'-shaped shape memory polymers
%
% Thomas Höft, University of St. Thomas
% hoft@stthomas.edu
% 
% last updated: 6 Aug 2025

% NOTEs: 
% *) requires Image Processing Toolbox; 
% *) requires Computer Vision Toolbox for saving image with overlay;
% *) Statistics and Machine Learning Toolbox required for fitting vertical
% lines -- or just rotate the image 

% If the region selection box disappears, type 
% >> clear
% at the command line to reset it (and weverything else too).

PINK = true; % true for pink-on-black, false for black-on-white
SAVE = true; % save trimmed image with overlaid lines and spreadsheet
ENDS = true; % use ends of filament as the straight lines we're finding 
             % the angle between -- set to false to automatically find the
             % segments with lowest curvature (i.e. straightest) 
EXTRA_DELETE = true; % selectable region to get rid of filament tail or mount
ROT_90 = false; % if you get a failed fit to a vertical line, rotate 90 degrees
LINE_COLOR = 'w'; % color of line on final image: 
                  % 'k' for black
                  % 'r' for red
                  % 'g' for green
                  % 'm' for magenta
                  % 'c' for cyan
                  % 'b' for blue
                  % 'w' for white
PLOTS = false; % show diagnostic plots

%#ok<*UNRCH> % don't warn on unreachable code

%% get filenames

% ABEER GOT SOME WEIRDNESS WITH "invalid directory to operate on"
% --> PUT IN A TRY/CATCH HERE?

if exist('pth', 'var') && any(pth~=0)
    [fn,pth] = uigetfile([pth '*.*'],'MultiSelect','on');
else
    [fn,pth] = uigetfile('*.*','MultiSelect','on');
end

if ~iscell(fn) % only one file, fn is string, convert to cell array
    fn = {fn};
    nfn = 1;
else
    nfn = length(fn);
end

if SAVE, angles_save = zeros(nfn,1); end % angle storage to write to csv

% start loop over all files
for fn_ii=1:nfn

%% load
disp("Processing " + fn_ii + " of " + nfn + ": " + fn{fn_ii});
imfn = [pth filesep fn{fn_ii}];

im_orig = imread(imfn);

if ROT_90, im_orig = rot90(im_orig,-1); end % rotates by 90 degrees, duh

figure(1); imagesc(im_orig); axis image; colormap gray; colorbar;
title(fn{fn_ii},'Interpreter','none')


%% color -> grayscale
if PINK
    im_gray = im_orig(:,:,1); % red channel has largest contrast
else
    im_gray = double(rgb2gray(im_orig))/255;
end

%% trim
% user selects sub-region to analyze
[Morig, Norig, ~] = size(im_orig);
if exist('roi_rect','var') % check if we've opened the window
    if ishandle(roi_rect) % check if window still open
        rect_pos = roi_rect.Position;
        rect_rot = roi_rect.RotationAngle;
    end
else
    rect_pos = [Norig/4,Morig/4,Norig/2,Morig/2];
    rect_rot = 0;
end
    
figure(1001); imagesc(im_orig); axis image; colormap gray; colorbar;
title('Select region of interest and press any key');
roi_rect = images.roi.Rectangle(gca,...
    'Position', rect_pos, 'RotationAngle', rect_rot,...
    'Rotatable', false, 'DrawingArea', 'unlimited');
addlistener(roi_rect,'ROIMoved',@allevents);
%disp('Select region of interest and press any key');
pause;

% WHAT HAPPENS IF RECTANGLE GOES OUTSIDE OF RANGE OF ARRAY? 
% SHOULD CHECK FOR THAT AND TRIM TO [0 N] and [0 M]

t_v = round(roi_rect.Vertices); % trim vertices 4x2 
                                % [x1,y1; x2,y2; x3,y3; x4,y4] 
                                % CCW around rect starting in U-L?
im_trim = im_gray(t_v(1,2):t_v(3,2), t_v(1,1):t_v(3,1)); 
im_orig_trim = im_orig(t_v(1,2):t_v(3,2), t_v(1,1):t_v(3,1),:); 

[M,N]=size(im_trim);

figure(2); imagesc(im_trim); axis image; colormap gray; colorbar;
title('trimmed grayscale image');

if EXTRA_DELETE
    % also have user select region to delete
    if exist('mount_rect','var') % check if we've opened the window
        if ishandle(mount_rect) % check if window still open
            % check if any rectangle corner is outside of view
            % (can happen if user selects new trim box)
            m_v = round(mount_rect.Vertices);
            if any(m_v<0,'all') || any(m_v>[N,M],'all') % reset position
                mount_pos = [N/8,M/8,N/6,M/6]; % [x_0, y_0, x_width, y_width]
                mount_rot = 0; 
            else % old position is still ok
                mount_pos = mount_rect.Position;
                mount_rot = mount_rect.RotationAngle;
            end
        end
    else
        mount_pos = [N/8,M/8,N/6,M/6]; % [x_0, y_0, x_width, y_width]
        mount_rot = 0;
    end
    figure(1002); imagesc(im_trim); axis image; colormap gray; colorbar;
    title('select region to delete and press any key');
    mount_rect = images.roi.Rectangle(gca,...
        'Position', mount_pos, 'RotationAngle', mount_rot,...
        'Rotatable', true, 'DrawingArea', 'auto');
    addlistener(mount_rect,'ROIMoved',@allevents);
    %disp('Select region to delete and press any key');
    pause;

    m_v = round(mount_rect.Vertices); % mount vertices 4x2 
                                      % [x1,y1; x2,y2; x3,y3; x4,y4] 
                                      % CCW around rect starting in U-L?
    im_del = zeros(size(im_trim));
    im_del(m_v(1,2):m_v(3,2), m_v(1,1):m_v(3,1)) = 1; 
end

%% --> bw
if PINK % pink on dark
    im_bw = imbinarize(im_trim, 0.8);
    % don't do hole fill-in b/c so much background need to erode first
else % black on white
    im_bw = imbinarize(1-im_trim, 0.8);
    % fill in any holes left from specular reflections:
    im_bw = imfill(im_bw,"holes"); % need to do this first for black filament
end

% delete mount
if EXTRA_DELETE, im_bw = max(im_bw - im_del,0); end 

% get rid of stray little bits
se = strel("disk",9);
im_bw = imerode(im_bw,se);

% fill in any holes left from specular reflections
im_bw = imfill(im_bw,"holes");

% smooth out boundary -- gets rid of wiggles/ripples from textured
% background (developed b/c of pink filament on lab bench, doesn't hurt
% black on white)
ker_d = 51; 
%ker = ones(ker_d)/ker_d^2; % mean
ker = exp(-(-(ker_d-1)/2:(ker_d-1)/2).^2); ker = ker/sum(ker); % gaussian
im_bw = imbinarize(conv2(im_bw,ker,'same')); % blur to smooth, re-binarize

% helps get rid of residual boundary weirdness from background and specular
% reflections: 
se1 = strel("disk",round(ker_d/4));
im_bw = imclose(im_bw, se1); % dilation followed by erosion

% get skeleton (bwskel)
im_skel = bwskel(logical(im_bw));% bwskel needs logical not double

if PLOTS
    figure(100); imagesc(im_bw); axis image; colormap gray; colorbar;
    figure(101); imagesc(im_bw+im_skel); axis image; colorbar;
end

%% extract centerline/bwskel of filament as parameterized curve
% get list of (x,y) co-ordinates of each pixel in in (regionprops PixelList)
s = regionprops(im_skel, 'PixelList'); % same as ind2sub([M,N],find(im_skel))
%s = regionprops(im_skel, 'PixelIdxList'); % same as find(im_skel(:))

if length(s)~=1 
    warning("More than one segmented region detected; keeping largest...")
    reg_len = cellfun('length',{s.PixelList}); % get lengths
    [~,idx] = sort(reg_len,"descend"); % sort so idx(1) points to longest
    s = s(idx(1)); % keep longest, discard rest
end

% turn in to a paramaterized curve    (is there an easier way?)
im_skel_tmp = im_skel;
x = s.PixelList(:,1); y = s.PixelList(:,2);
npix = size(s.PixelList,1); % number of pixels in the curve
tmp_crv_1 = zeros(size(s.PixelList));
tmp_crv_2 = zeros(size(s.PixelList));
% first point
tmp_crv_1(1,:) = s.PixelList(1,:);
tmp = im_skel(y(1)-1:y(1)+1, x(1)-1:x(1)+1); 
tmp(5) = 0; % current point --> exclude
nbr_idx = find(tmp,1); % index of one of the two neighbors
[nbr_y, nbr_x] = ind2sub([3,3],nbr_idx); % subs of one of the two neighbors
new_xy = [x(1),y(1)]+([nbr_x,nbr_y]-[2,2]); % (x,y) of new pt to add
tmp_crv_1(2,:) = new_xy; % add either of nbr pixels to list
new_xy = s.PixelList(1,:);
for ii=1:npix-1
    old_xy = new_xy; % current point
    im_skel_tmp(old_xy(2),old_xy(1)) = 0; % exlude current point
    tmp = im_skel_tmp(old_xy(2)-1:old_xy(2)+1, old_xy(1)-1:old_xy(1)+1); % extract neighborhood
    nbr_idx = find(tmp,1); % index of nbr (one of the two neighbors if ii=1)
    if isempty(nbr_idx) % no neighbor --> end of curve --> break
        break;
    end
    [nbr_y, nbr_x] = ind2sub([3,3],nbr_idx); % subs of one of the two neighbors
    new_xy = [old_xy(1),old_xy(2)]+([nbr_x,nbr_y]-[2,2]); % (x,y) of new pt to add
    tmp_crv_1(ii+1,:) = new_xy; % add nbr pixel to list
end
crv_1_end = ii;
% now do other half of curve if there's still points left to process
if crv_1_end < npix-1
    new_xy = s.PixelList(1,:);
    for ii=1:npix-1
        old_xy = new_xy; % current point
        im_skel_tmp(old_xy(2),old_xy(1)) = 0; % exlude current point
        tmp = im_skel_tmp(old_xy(2)-1:old_xy(2)+1, old_xy(1)-1:old_xy(1)+1); % extract neighborhood
        nbr_idx = find(tmp,1); % index of nbr (one of the two neighbors if ii=1)
        if isempty(nbr_idx) % no neighbor --> end of curve --> break
            break;
        end
        [nbr_y, nbr_x] = ind2sub([3,3],nbr_idx); % subs of one of the two neighbors
        new_xy = [old_xy(1),old_xy(2)]+([nbr_x,nbr_y]-[2,2]); % (x,y) of new pt to add
        tmp_crv_2(ii,:) = new_xy; % add nbr pixel to list
    end
    crv_2_end = ii-1;
    crv_xy = [flipud(tmp_crv_2(1:crv_2_end,:)); tmp_crv_1(1:crv_1_end,:)];
else % got all the points, so just save it
    crv_xy = tmp_crv_1(1:crv_1_end,:);
end

if PLOTS % double-check that curve is correct;
    tmp_im = zeros(M,N); tmp_im(sub2ind([M,N],crv_xy(:,2),crv_xy(:,1))) = 1;
    figure(110); imagesc(tmp_im); axis image; colormap gray; colorbar;
end

%% separate into flat & curved: 
x = crv_xy(:,1); y = crv_xy(:,2); 
crv_len = length(x);

if PLOTS | ~ENDS % either for use in computation or for diagnostic display
    %  compute curvature at each point
    wid = round(crv_len/4); % smoothing window width
    x = smooth(x,wid,'loess');
    dx = gradient(x); dx = smooth(dx,wid,'loess');
    ddx = gradient(dx); ddx = smooth(ddx,wid,'loess');
    y = smooth(y,wid,'loess');
    dy = gradient(y); dy = smooth(dy,wid,'loess');
    ddy = gradient(dy); ddy = smooth(ddy,wid,'loess');

    k = (dx.*ddy-dy.*ddx)./(dx.^2+dy.^2).^1.5;
    k = abs(k);
    k = smooth(k,wid);
end

if ~ENDS % find flat regions via curvature
    % expect a W shape (or just a single bump in middle), so find two minima
    % the max inbetween helps find the thresholds we want
    crv_mid = round(crv_len/2);
    [min1,id1] = min(k(10:crv_mid-1)); % ignore first 9 pixels
    id1 = id1 + 10 - 1; % shift
    [min2,id2] = min(k(crv_mid+1:end-10+1)); % ignore last 9 pixels
    id2 = id2 + crv_mid + 1 - 1; % shift
    [maxmid,idmaxmid] = max(k(id1:id2));
    idmaxmid = idmaxmid + id1 - 1; % shift
    kap_thresh1 = (maxmid-min1)/4 + min1; % 1/4 of height from min to max
    kap_thresh2 = (maxmid-min2)/4 + min2;
    kap_zer = false(size(k));
    kap_zer(1:idmaxmid) = k(1:idmaxmid)<kap_thresh1; % keep low-curvature pts
    kap_zer(idmaxmid:end) = k(idmaxmid:end)<kap_thresh2;
    if idmaxmid==id1 % curvature is monotone decreasing L->Mid -- straight?
        kap_zer(1:round(crv_len/8)) = true; % keep left 1/8th?
    end
    if idmaxmid==id2 % curvature is monotone decreasing Mid->R -- straight?
        kap_zer(end-round(crv_len/8):end) = true; % keep right 1/8th?
    end
end

if ENDS % just keep L & R 1/8th
    kap_zer = false(crv_len,1);
    % length/6 seems to give enough points for a good fit to the line
    % leave off first & last length/6 to avoid weirness at ends of curve
    seg_1_idx = round(crv_len/32):round(crv_len/6);
    seg_2_idx = (crv_len-round(crv_len/6)):(crv_len-round(crv_len/32));
    kap_zer(seg_1_idx) = true;
    kap_zer(seg_2_idx) = true;
end

if PLOTS
    figure(121); plot(x); title('x'); 
    figure(122); plot(dx); title('dx'); 
    figure(123); plot(ddx); title('ddx');
    figure(124); plot(y); title('y');
    figure(125); plot(dy); title('dy'); 
    figure(126); plot(ddy); title('ddy');
    if ENDS
        figure(127); plot(1:crv_len,k)
        title("|\kappa|");
    else
        figure(127); plot(1:crv_len,k, ...
                         [1,idmaxmid],kap_thresh1*ones(1,2), ...
                         [idmaxmid,crv_len],kap_thresh2*ones(1,2), ...
                          id1,min1,'ko', id2,min2,'ko', idmaxmid,maxmid,'ko');
        title("|\kappa|");
        ylim([0,max(k(wid+1:end-wid))*1.1])
    end

    tmp_idx = sub2ind([M,N],crv_xy(kap_zer,2),crv_xy(kap_zer,1));
    tmp_im2 = zeros(M,N); tmp_im2(tmp_idx) = 1;
    figure(130); imagesc(tmp_im2+tmp_im); axis image; colorbar;
end

% TODO: I THINK ALL OF THIS IS IRRELEVANT...?
if ~ENDS
    %  pull out the two sections from above with small curvature
    cc = bwconncomp(kap_zer);
    if cc.NumObjects~=2
        %error("Fewer than two straight segments detected.");
        %warning("Number of straight segments detected is " + cc.NumObjects + "; should be TWO.");
        if cc.NumObjects>2
            %seg_len = cellfun('length',cc.PixelIdxList);
            %[~,idx] = sort(seg_len,"descend");
            %cc.PixelIdxList = cc.PixelIdxList(idx);

            for ii=1:cc.NumObjects
                if any(cc.PixelIdxList{ii}==id1) % this one has segment 1
                    seg_1_idx = cc.PixelIdxList{ii};
                end
                if any(cc.PixelIdxList{ii}==id2) % this one has segment 1
                    seg_2_idx = cc.PixelIdxList{ii};
                end
            end
        else
            warning("Only one straight line segment detected");
            % TODO
            % IF ONLY ONE SEGMENT, CUT IT SOMEWHERE (HALFWAY?) AND FIT TWO LINES?
            % OR ASSUME ANGLE = 0 ?
        end
    else
        seg_1_idx = cc.PixelIdxList{1};
        seg_2_idx = cc.PixelIdxList{2};
    end
end

seg_1_len = length(seg_1_idx); 
seg_2_len = length(seg_2_idx);

% fit line to flat sections
%  segment 1
X1 = ones(seg_1_len,2);
X1(:,2) = crv_xy(seg_1_idx,1); % data matrix -- predictor
Y1 = crv_xy(seg_1_idx,2); % response
beta_1 = (X1'*X1)\X1'*Y1; % linear least squares
%  segment 2
X2 = ones(seg_2_len,2);
X2(:,2) = crv_xy(seg_2_idx,1); % data matrix -- predictor
Y2 = crv_xy(seg_2_idx,2); % response
beta_2 = (X2'*X2)\X2'*Y2; % linear least squares

% check to see if fit to vertical line and it failed
r1s = sign(Y1-X1*beta_1); % sign of residual
r2s = sign(Y2-X2*beta_2);
if all(r1s(1:round(length(r1s)/8)) == -r1s(end-round(length(r1s)/8)+1:end))
    % sign of residual at one end of line is uniformly opposite of other
    % indication of heteroskedacicity, which can happen when fitting
    % near-vertical line
    warning("Possible failed fit to vertical line -- check and if so enable ROT90.")
    % try total least squares (orthogonal distance regression)
    beta_1 = tls(X1(:,2), Y1); % function defined at bottom of this file
end
if all(r2s(1:round(length(r2s)/8)) == -r2s(end-round(length(r2s)/8)+1:end))
    % repeat for other line
    warning("Possible failed fit to vertical line -- check and if so enable ROT90.")
    beta_2 = tls(X2(:,2), Y2); 
end

% get (x,y) coords of lines at edges of image (will use for plotting)
xrng1 = [round(min( max(-beta_1(1)/beta_1(2),0),    N)) ...
         round(max( min((M-beta_1(1))/beta_1(2),N), 0))];
yrng1 = beta_1(1) + beta_1(2)*xrng1;

xrng2 = [round(min( max(-beta_2(1)/beta_2(2),0), N))
         round(max( min((M-beta_2(1))/beta_2(2),N), 0))];
yrng2 = beta_2(1) + beta_2(2)*xrng2;

if PLOTS
    % plot regression line for x & y values in image
    figure(128); plot(X1(:,2),Y1,'b.', xrng1,yrng1,'r'); hold on
    title("segment 1 fit line, y = " + round(beta_1(1)) + " + " + round(beta_1(2)) + " x");
    axis([0,N,0,M]);

    figure(128); plot(X2(:,2),Y2,'b.', xrng2,yrng2,'r', crv_xy(:,1),crv_xy(:,2)); 
    title({gca().Title.String, "segment 2 fit line, y = " + round(beta_2(1)) + " + " + round(beta_2(2)) + " x"});
    axis([0,N,0,M]); hold off;
end

%% compute angle
% I'm ignoring whether it should be acute/obtuse 
% -- that'll have to be done manually after-the-fact
theta = atan(abs((beta_1(2)-beta_2(2))/(1+beta_1(2)*beta_2(2))))*180/pi;
disp("                   " + theta + " degrees");
if SAVE, angles_save(fn_ii) = theta; end

% draw on photo
figure(3); imagesc(im_orig_trim); axis image; axis off;
hold on; plot(xrng1,yrng1, LineWidth=2); plot(xrng2,yrng2, LineWidth=2);
plot(X1(:,2), Y1, 'b', LineWidth=4);
plot(X2(:,2), Y2, 'r', LineWidth=4);
hold off; axis([0,N,0,M]); title("angle = " + theta + " degrees")

%% save image
if SAVE
    % start with trimmed image and draw lines on it (no other annotation)
    im_save = im_orig_trim;
    % plot just the portion of the line in the range of pixels in the image
    im_save = insertShape(im_save, 'Line', ...
                          [[xrng1(1),yrng1(1)], ...
                           [xrng1(2),yrng1(2)]], ...
                          'Color', LINE_COLOR, 'LineWidth', 8);
    im_save = insertShape(im_save, 'Line', ...
                          [[xrng2(1),yrng2(1)], ...
                           [xrng2(2),yrng2(2)]], ...
                          'Color', LINE_COLOR, 'LineWidth', 8);

    figure(4); imshow(im_save); 

    [~,fn_tmp,~] = fileparts(fn{fn_ii}); % get filename, strip extension
    % usefilename with explicit path
    imwrite(im_save, [pth, filesep, fn_tmp] + "_angle" + ".png");
end

%% end loop over all files
end

%% save angles to csv and xlsx
if SAVE
    % ang_fn = pth + "angles.csv";
    % writecell({pth}, ang_fn)
    % T = table(fn', angles_save, 'VariableNames', {'fileName', 'angle'});
    % writetable(T, ang_fn, 'WriteMode','Append');

    ang_fn = pth + "angles.xlsx";
    T = table(fn', angles_save, 'VariableNames', {'fileName', 'angle'});
    writetable(T, ang_fn, 'WriteMode','replacefile');
    writecell({pth}, ang_fn, 'WriteMode','Append')
end


%% HELPER FUNCTIONS BELOW THIS LINE

%% Total Least Squares / Orthogonal Distance Regression

function beta = tls(x,y)
    % fit y = beta(1) + beta(2) x,  where the line is close to vertical
    % see https://stats.stackexchange.com/a/136597 or wikipedia
    x = x(:); y = y(:); % insist on column vectors
    x_bar = mean(x); y_bar = mean(y);
    v = pca([x-x_bar,y-y_bar]); % PCA on zero-center data
    beta(2) = v(2,1)/v(1,1); % slope
    beta(1) = y_bar - beta(2)*x_bar; % restore y-intercept (de-center)
end


%%
function ret = smooth_th(arg, wid, fil) %#ok<DEFNU>
    % use an odd number for wid
    arguments
        arg
        wid
        fil = 'mean';
    end
    arg = arg(:); % make sure it's a column
    switch fil
        case 'mean'
            ker = ones(wid,1)/wid;
        case 'gauss'
            ker = exp(-(-(wid-1)/2:(wid-1)/2).^2); ker = ker/sum(ker); 
    end
    ret = conv([fliplr(arg(1:(wid-1)/2)); arg; fliplr(arg(end-(wid-1)/2+1:end))], ker, 'valid');
end


%%
function allevents(~, ~, ~) % dummy to catch the trim rectangle move event
end
