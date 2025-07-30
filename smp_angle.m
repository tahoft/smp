%% load
%#ok<*UNRCH>

fn = 'IMG_8776.jpeg';
im_orig = imread(fn);
figure(1); imagesc(im_orig); axis image; colormap gray; colorbar;

PINK = true; 
SAVE = true;
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
figure(2002); imagesc(im_trim); axis image; colormap gray; colorbar;
title('trimmed image');

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

%% correct for varied gray background

im_flat = imflatfield(im_trim,30);
figure(3); imagesc(im_flat); axis image; colormap gray; colorbar;

% smooth to get rid of texture on lab bench
im_smooth = imgaussfilt(im_flat,10);
figure(4); imagesc(im_smooth); axis image; colormap gray; colorbar;

% texture background removal
%im_notex = entropyfilt(im_flat, true(3)); % remove texture
%im_notex = imflatfield(im_notex, 50);
%im_notex = imgaussfilt(im_notex, 20); % smooth
im_notex = stdfilt(im_flat, true(11)); % remove texture
im_notex = (im_notex-min(im_notex(:)))/(max(im_notex(:))-min(im_notex(:))); % make sure it's [0,1]
figure(5); imagesc(1-im_notex); axis image; colormap gray; colorbar;


%% --> bw

%T = adaptthresh(1-im_trim, 'NeighborhoodSize', 9);
%im_bw = imbinarize(1-im_trim, T);
im_bw = imbinarize(im_trim, 0.8);
%T = adaptthresh(1-im_smooth, 0.6, 'NeighborhoodSize', 9);
%im_bw = imbinarize(1-im_smooth, T);
%im_bw = imbinarize(im_smooth, 0.63);
%im_bw = imbinarize(1-im_notex, 0.85);

% delete mount
if MOUNT_DELETE, im_bw = max(im_bw - im_mount,0); end 


se = strel("disk",9);
im_bw = imerode(im_bw,se);
se = strel("disk",25);
im_bw = imdilate(im_bw,se);
im_bw = imerode(im_bw,se);
im_bw = imdilate(im_bw,se);
im_bw = imerode(im_bw,se);
im_bw = imfill(im_bw,"holes");

figure(2); imagesc(im_bw); axis image; colormap gray; colorbar;

% get skeleton (bwskel)
im_skel = bwskel(logical(im_bw));% bwskel needs logical not double
figure(21); imagesc(im_bw+im_skel); axis image; colorbar;


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
tmp_im = zeros(M,N); tmp_im(sub2ind([M,N],crv_xy(:,2),crv_xy(:,1))) = 1;
figure(701); imagesc(tmp_im); axis image; colormap gray; colorbar;


%% separate into flat & curved: 

%  compute curvature at each point
x = crv_xy(:,1); y = crv_xy(:,2); 
crv_len = length(x);
%wid = 80;
%wid = 100;
wid = crv_len/8;
%x = smoothdata(x,"movmean",wid); y = smoothdata(y,"movmean",wid);
%x = smooth(x, 21);
dx = gradient(x); % central diff in middle, forward diff at ends
dx = smoothdata(dx,"movmean",wid); 
%dx = smooth(dx,wid); 
dy = gradient(y);
dy = smoothdata(dy,"movmean",wid);
ddx = gradient(dx); % 2nd order central diff
ddx = smoothdata(ddx,"movmean",wid); 
%ddx = smooth(ddx,wid);
ddy = gradient(dy); % 2nd order central diff
ddy = smoothdata(ddy,"movmean",wid);

wid = round(crv_len/16);
x = smooth(x,wid);
dx = gradient(x);
dx = smooth(dx,2*wid);
ddx = gradient(dx);
ddx = smooth(ddx,wid);
y = smooth(y,wid);
dy = gradient(y);
dy = smooth(dy,wid);
ddy = gradient(dy);
ddy = smooth(ddy,wid);


k = (dx.*ddy-dy.*ddx)./(dx.^2+dy.^2).^1.5;
k = abs(k);
%k = smoothdata(k,"movmedian",11);
%k = smoothdata(k,"movmean",wid);
k = smooth(k,wid);


% trim off first & last wid to avoid any weirdness there from smoothing
% window
%n = size(crv_xy,1);
%k = k(wid+1:end-wid,:);



ddy_nonz = ddx~=0;
%CURVATURE_THRESH = max(k)/2;
%CURVATURE_THRESH = max(k(wid+1:end-wid-1))/4;
CURVATURE_THRESH = (max(k(wid+1:end-wid-1))-min(k(wid+1:end-wid-1)))/4 + min(k(wid+1:end-wid-1));
%kap_zer(wid+1:end-wid) = k < CURVATURE_THRESH;
kap_zer = k < CURVATURE_THRESH;

% expect a W shape, so find two minima
% the max inbetween helps find the thresholds we want
crv_mid = round(crv_len/2);
[min1,id1] = min(k(10:crv_mid));
id1 = id1 + 10; % shift 
[min2,id2] = min(k(crv_mid+1:end-10));
id2 = id2 + crv_mid + 1; % shift 
[maxmid,idmaxmid] = max(k(id1:id2));
idmaxmid = idmaxmid + id1 + 1; % shift 
kap_thresh1 = (maxmid-min1)/2 + min1;
kap_thresh2 = (maxmid-min2)/2 + min2;
kap_zer = false(size(k));
kap_zer(1:idmaxmid) = k(1:idmaxmid)<kap_thresh1;
kap_zer(idmaxmid:end) = k(idmaxmid:end)<kap_thresh2;


if PLOTS
    figure(101); plot(x); title('x'); figure(102); plot(dx); title('dx'); figure(103); plot(ddx); title('ddx');
    figure(104); plot(y); title('y');figure(105); plot(dy); title('dy'); figure(106); plot(ddy); title('ddy');
    %figure(107); plot(1:length(k),k, [1,length(k)],CURVATURE_THRESH*ones(1,2)); title("|\kappa|");
    figure(107); plot(1:crv_len,k, ...
                      [1,idmaxmid],kap_thresh1*ones(1,2), ...
                      [idmaxmid,crv_len],kap_thresh2*ones(1,2), ...
                      id1,min1,'ko', id2,min2,'ko', idmaxmid,maxmid,'ko'); 
    title("|\kappa|");
    ylim([0,max(k(wid+1:end-wid))*1.1])
    %figure(108); plot(ddx.*ddy); title('prod of concavities');
end

tmp_idx = sub2ind([M,N],crv_xy(kap_zer,2),crv_xy(kap_zer,1));
tmp_im2 = zeros(M,N); tmp_im2(tmp_idx) = 1;
figure(702); imagesc(tmp_im2+tmp_im); axis image; colorbar;

drawnow; 

%  pull out two longest sections with abs(kappa) > thresh
cc = bwconncomp(kap_zer);
% IF ONLY ONE SEGMENT, CUT IT SOMEWHERE (HALFWAY?) AND FIT TWO LINES?
% OR ASSUME ANGLE = 0 ? 
if cc.NumObjects~=2
    %error("Fewer than two straight segments detected.");
    warning("Number of straight segments detected is " + cc.NumObjects + "; should be TWO.");
    if cc.NumObjects>2
        seg_len = cellfun('length',cc.PixelIdxList);
        [~,idx] = sort(seg_len,"descend");
        cc.PixelIdxList = cc.PixelIdxList(idx);
    end
end
% ALSO NEED TO HANDLE IF MORE THAN TWO SEGMENTS -- TAKE LONGEST TWO?
seg_1_idx = cc.PixelIdxList{1}; seg_1_len = length(seg_1_idx);
seg_2_idx = cc.PixelIdxList{2}; seg_2_len = length(seg_2_idx);


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
    figure(703); plot(X(:,2),Y,'b.', x_1,y_1,'r'); hold on
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
    figure(703); plot(X(:,2),Y,'b.', x_2,y_2,'r', crv_xy(:,1),crv_xy(:,2)); 
    title({gca().Title.String, "segment 2 fit line, y = " + round(beta_2(1)) + " + " + round(beta_2(2)) + " x"});
    axis([0,N,0,M]); hold off;
end

%%
% compute angle
theta = atan(abs((beta_1(2)-beta_2(2))/(1+beta_1(2)*beta_2(2))))*180/pi;

% draw on photo
figure(6); imagesc(im_orig_trim); axis image; axis off;
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

    figure(2009); imshow(im_save);

    [fn_pth,fn_name,~] = fileparts(fn);
    imwrite(im_save, fn_name + "_angle_" + ...
        floor(theta) + "_" + round((theta-floor(theta))*100) + ".png");
    % may want to add path (may change with gui to select images)
end


%%
function ret = smooth_th(arg,wid)
    ret = conv([fliplr(arg(1:(wid-1)/2)); arg; fliplr(arg(end-(wid-1)/2+1:end))], ones(wid,1)/wid, 'valid');
end


%%
function allevents(~, ~, ~) % dummy to catch the trim rectangle move event
end
