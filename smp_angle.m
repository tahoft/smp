%% smp_angle
% Extract angle of 'U'-shaped shape memory polymers
% For Brittany Nelson-Cheeseman, Elsie Kmecak, Abeer Najajra
%
% by Thomas Höft 

%% load
%#ok<*UNRCH>

fn = 'IMG_8776.jpeg';
[fn_pth,fn_name,~] = fileparts(fn);

im_orig = imread(fn);
im_orig = rot90(im_orig,-1);
figure(1); imagesc(im_orig); axis image; colormap gray; colorbar;

PINK = true; 
SAVE = false;
PLOTS = true;
MOUNT_DELETE = false; % selectable region to get rid of black mount
LINE_COLOR = 'w'; % color of line on final image: 
                  % 'k' for black
                  % 'r' for red
                  % 'g' for green
                  % 'm' for magenta
                  % 'c' for cyan
                  % 'b' for blue
                  % 'w' for white
%CURVATURE_THRESH = 0.005; % unused...


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
title('Select region and press any key');
roi_rect = images.roi.Rectangle(gca,...
    'Position', rect_pos, 'RotationAngle', rect_rot,...
    'Rotatable', false, 'DrawingArea', 'unlimited');
addlistener(roi_rect,'ROIMoved',@allevents);
disp('Select region and press any key');
pause;

trim_info = round(roi_rect.Position);
% using roi_rect.Vertices might be easier
im_trim = im_gray(trim_info(2):trim_info(2)+trim_info(4), ...
    trim_info(1):trim_info(1)+trim_info(3));
im_orig_trim = im_orig(trim_info(2):trim_info(2)+trim_info(4), ...
    trim_info(1):trim_info(1)+trim_info(3),:);
[M,N]=size(im_trim);
figure(2); imagesc(im_trim); axis image; colormap gray; colorbar;
title('trimmed grayscale image');

if MOUNT_DELETE
    % also have user select region to get rid of any mounts
    if exist('mount_rect','var') % check if we've opened the window
        if ishandle(mount_rect) % check if window still open
            mount_pos = mount_rect.Position;
            mount_rot = mount_rect.RotationAngle;
        end
    else
        mount_pos = [N/8,M/8,N/6,M/6];
        mount_rot = 0;
    end
    figure(1002); imagesc(im_trim); axis image; colormap gray; colorbar;
    title('select mount delete region and press any key');
    mount_rect = images.roi.Rectangle(gca,...
        'Position', mount_pos, 'RotationAngle', mount_rot,...
        'Rotatable', false, 'DrawingArea', 'auto', ...
        'Label', 'mount delete');
    addlistener(mount_rect,'ROIMoved',@allevents);
    disp('Select region to delete mount and press any key');
    pause;
    im_mount = zeros(size(im_trim));
    abcd = round(mount_rect.Position);
    im_mount(abcd(2):abcd(2)+abcd(4)-1, abcd(1):abcd(1)+abcd(3)-1) = 1;
end

%% --> bw

im_bw = imbinarize(im_trim, 0.8);

% delete mount
if MOUNT_DELETE, im_bw = max(im_bw - im_mount,0); end 

% get rid of stray little bits
se = strel("disk",9);
im_bw = imerode(im_bw,se);

% fill in any holes left from specular reflections
im_bw = imfill(im_bw,"holes");

% smooth out boundary 
ker_d = 51; 
%ker = ones(ker_d)/ker_d^2; % mean
ker = exp(-(-(ker_d-1)/2:(ker_d-1)/2).^2); ker = ker/sum(ker); % gaussian
im_bw = imbinarize(conv2(im_bw,ker,'same')); % blur to smooth
se1 = strel("disk",round(ker_d/4));

% helps get rid of residual boundary weirdness from background and specular
% reflections: 
im_bw = imclose(im_bw, se1); % dilation followed by erosion

if PLOTS
    figure(100); imagesc(im_bw); axis image; colormap gray; colorbar;
end

% get skeleton (bwskel)
im_skel = bwskel(logical(im_bw));% bwskel needs logical not double
if PLOTS
    figure(101); imagesc(im_bw+im_skel); axis image; colorbar;
end

%%
% get list of (x,y) co-ordinates of each pixel in in (regionprops PixelList)
s = regionprops(im_skel, 'PixelList'); % same as ind2sub([M,N],find(im_skel))
%s = regionprops(im_skel, 'PixelIdxList'); % same as find(im_skel(:))

if length(s)~=1 
    warning("More than one region detected; keeping largest...")
    reg_len = cellfun('length',{s.PixelList});
    [~,idx] = sort(reg_len,"descend");
    s = s(idx(1));
end

% turn in to a paramaterized curve
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

% trim off first & last 1/16th to avoid any weirdness there
%n = size(crv_xy,1);
%crv_xy = crv_xy(round(n/16):end-round(n/16),:);

% double-check that curve is correct;
if PLOTS
    tmp_im = zeros(M,N); tmp_im(sub2ind([M,N],crv_xy(:,2),crv_xy(:,1))) = 1;
    figure(110); imagesc(tmp_im); axis image; colormap gray; colorbar;
end

%% separate into flat & curved: 

%  compute curvature at each point
x = crv_xy(:,1); y = crv_xy(:,2); 
crv_len = length(x);

wid = round(crv_len/4); 
%wid = 51;
x = smooth(x,wid,'loess');
dx = gradient(x);
dx = smooth(dx,wid,'loess');
ddx = gradient(dx);
ddx = smooth(ddx,wid,'loess');
y = smooth(y,wid,'loess');
dy = gradient(y);
dy = smooth(dy,wid,'loess');
ddy = gradient(dy);
ddy = smooth(ddy,wid,'loess');


k = (dx.*ddy-dy.*ddx)./(dx.^2+dy.^2).^1.5;
k = abs(k);
k = smooth(k,wid);

% expect a W shape, so find two minima
% the max inbetween helps find the thresholds we want
crv_mid = round(crv_len/2);
[min1,id1] = min(k(10:crv_mid));
id1 = id1 + 10; % shift 
[min2,id2] = min(k(crv_mid+1:end-10));
id2 = id2 + crv_mid + 1; % shift 
[maxmid,idmaxmid] = max(k(id1:id2));
idmaxmid = idmaxmid + id1 + 1; % shift 
kap_thresh1 = (maxmid-min1)/4 + min1;
kap_thresh2 = (maxmid-min2)/4 + min2;
kap_zer = false(size(k));
kap_zer(1:idmaxmid) = k(1:idmaxmid)<kap_thresh1;
kap_zer(idmaxmid:end) = k(idmaxmid:end)<kap_thresh2;


if PLOTS
    figure(121); plot(x); title('x'); 
    figure(122); plot(dx); title('dx'); 
    figure(123); plot(ddx); title('ddx');
    figure(124); plot(y); title('y');
    figure(125); plot(dy); title('dy'); 
    figure(126); plot(ddy); title('ddy');
    figure(127); plot(1:crv_len,k, ...
                      [1,idmaxmid],kap_thresh1*ones(1,2), ...
                      [idmaxmid,crv_len],kap_thresh2*ones(1,2), ...
                      id1,min1,'ko', id2,min2,'ko', idmaxmid,maxmid,'ko'); 
    title("|\kappa|");
    ylim([0,max(k(wid+1:end-wid))*1.1])

    tmp_idx = sub2ind([M,N],crv_xy(kap_zer,2),crv_xy(kap_zer,1));
    tmp_im2 = zeros(M,N); tmp_im2(tmp_idx) = 1;
    figure(130); imagesc(tmp_im2+tmp_im); axis image; colorbar;
end

%  pull out the two sections from above with small curvatur
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
seg_1_len = length(seg_1_idx); 
seg_2_len = length(seg_2_idx);

% fit line to flat sections
%  segment 1
X = ones(seg_1_len,2);
X(:,2) = crv_xy(seg_1_idx,1); % data matrix -- predictor
Y = crv_xy(seg_1_idx,2); % response
beta_1 = (X'*X)\X'*Y; % linear least squares
x_1 = 1:N; y_1 = beta_1(1) + beta_1(2)*x_1; % regression line
x_1_data = X(:,2); 
y_1_data = Y; 
if PLOTS
    figure(128); plot(X(:,2),Y,'b.', x_1,y_1,'r'); hold on
    title("segment 1 fit line, y = " + round(beta_1(1)) + " + " + round(beta_1(2)) + " x");
    axis([0,N,0,M]);
end
%  segment 2
X = ones(seg_2_len,2);
X(:,2) = crv_xy(seg_2_idx,1); % data matrix -- predictor
Y = crv_xy(seg_2_idx,2); % response
beta_2 = (X'*X)\X'*Y; % linear least squares
x_2 = 1:N; y_2 = beta_2(1) + beta_2(2)*x_2; % regression line
x_2_data = X(:,2); 
y_2_data = Y; 
if PLOTS
    figure(128); plot(X(:,2),Y,'b.', x_2,y_2,'r', crv_xy(:,1),crv_xy(:,2)); 
    title({gca().Title.String, "segment 2 fit line, y = " + round(beta_2(1)) + " + " + round(beta_2(2)) + " x"});
    axis([0,N,0,M]); hold off;
end

%%
% compute angle
theta = atan(abs((beta_1(2)-beta_2(2))/(1+beta_1(2)*beta_2(2))))*180/pi;
disp("File " + fn_name + " angle: " + theta + " degrees");

% draw on photo
figure(3); imagesc(im_orig_trim); axis image; axis off;
hold on; plot(x_1,y_1, LineWidth=2); plot(x_2,y_2, LineWidth=2);
plot(x_1_data, y_1_data, 'b', LineWidth=4);
plot(x_2_data, y_2_data, 'r', LineWidth=4);
hold off; axis([0,N,0,M]); title("angle = " + theta + " degrees")

%% save image 
if SAVE
    im_save = im_orig_trim;
    im_save = insertShape(im_save, 'Line', ...
                          [[x_1(1),y_1(1)], [x_1(end),y_1(end)]], ...
                          'Color', LINE_COLOR, 'LineWidth', 8);
    im_save = insertShape(im_save, 'Line', ...
                          [[x_2(1),y_2(1)], [x_2(end),y_2(end)]], ...
                          'Color', LINE_COLOR, 'LineWidth', 8);

    figure(4); imshow(im_save);

    imwrite(im_save, fn_name + "_angle_" + ...
        floor(theta) + "_" + round((theta-floor(theta))*100) + ".png");
    % may want to add path (may change with gui to select images)
end

%% HELPER FUNCTIONS BELOW THIS LINE

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
