%% load
%#ok<*UNRCH>

%im_orig = imread('im1.png');
im_orig = imread('IMG_8776.jpeg');
figure(1); imagesc(im_orig); axis image; colormap gray; colorbar;

PINK = true; 
MOUNT_DELETE = false; % selectable region to get rid of black mount
%CURVATURE_THRESH = 0.005; % im1
%CURVATURE_THRESH = 0.0005; % im5

%% grayscale
if PINK
    im_gray = im_orig(:,:,1); % red channel has largest contrast
else
    im_gray = double(rgb2gray(im_orig))/255;
end

%% trim
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
title('select rectangular ROI');%, can close figure or leave open');
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
    % ALSO DO MOUNT REMOVAL
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
    title('select mount delete ROI');%, can close figure or leave open');
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
im_bw = logical(im_bw); % bwskel needs logical not double

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
im_skel = bwskel(im_bw);
figure(21); imagesc(im_bw+im_skel); axis image; colorbar;


%%
% get list of (x,y) co-ordinates of each pixel in in (regionprops PixelList)
s = regionprops(im_skel, 'PixelList'); % same as ind2sub([M,N],find(im_skel))
%s = regionprops(im_skel, 'PixelIdxList'); % same as find(im_skel(:))

% NEEd TO CHECK IF ONLY ONE REGION
if length(s)~=1
    warning("More than one region detected: continue at your own risk...")
end
s = s(1); 

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

% double-check that curve is correct;
tmp_im = zeros(M,N); tmp_im(sub2ind([M,N],crv_xy(:,2),crv_xy(:,1))) = 1;
figure(701); imagesc(tmp_im); axis image; colormap gray; colorbar;


%% separate into flat & curved: 

%  compute curvature at each point
x = crv_xy(:,1); y = crv_xy(:,2); 
%wid = 80;
%wid = 100;
wid = length(x)/8;
%x = smoothdata(x,"movmean",wid); y = smoothdata(y,"movmean",wid);
%x = smoothdata(x,"gaussian",40); y = smoothdata(y,"gaussian",40);
%dx = [x(2)-x(1); (x(3:end)-x(1:end-2))/2; x(end)-x(end-1)];
dx = gradient(x); % central diff in middle, forward diff at ends
dx = smoothdata(dx,"movmean",wid); 
dy = gradient(y);
dy = smoothdata(dy,"movmean",wid);
ddx = gradient(dx); % 2nd order central diff
ddx = smoothdata(ddx,"movmean",wid); 
ddy = gradient(dy); % 2nd order central diff
ddy = smoothdata(ddy,"movmean",wid);
k = (dx.*ddy-dy.*ddx)./(dx.^2+dy.^2).^1.5;
k = abs(k);
%k = smoothdata(k,"movmedian",11);
k = smoothdata(k,"movmean",wid);

ddy_nonz = ddx~=0;
CURVATURE_THRESH = max(k)/2;
kap_zer = k < CURVATURE_THRESH;

figure(101); plot(x); title('x'); figure(102); plot(dx); title('dx'); figure(103); plot(ddx); title('ddx');
figure(104); plot(y); title('y');figure(105); plot(dy); title('dy'); figure(106); plot(ddy); title('ddy');
figure(107); plot(1:length(k),k, [1,length(k)],CURVATURE_THRESH*ones(1,2)); title("|\kappa|");
%figure(108); plot(ddx.*ddy); title('prod of concavities');

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
    error("Number of straight segments detected is different than TWO.");
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
figure(703); plot(X(:,2),Y,'.', x_1,y_1); axis([0,N,0,M]);
title("segment 1 fit line, y = " + beta_1(1) + " + " + beta_1(2) + " x"); 
%  segment 2
X = ones(seg_2_len,2);
X(:,2) = crv_xy(seg_2_idx,1); % data matrix -- predictor
Y = crv_xy(seg_2_idx,2); % response
beta_2 = (X'*X)\X'*Y; % linear least squares
x_2 = 1:N; y_2 = beta_2(1) + beta_2(2)*x_2; % regression line
figure(704); plot(X(:,2),Y,'.', x_2,y_2); axis([0,N,0,M]);
title("segment 2 fit line, y = " + beta_2(1) + " + " + beta_2(2) + " x"); 

%%
% compute angle
theta = atan(abs((beta_1(2)-beta_2(2))/(1+beta_1(2)*beta_2(2))))*180/pi;

% draw on photo
figure(6); imagesc(im_orig_trim); axis image; axis off;
hold on; plot(x_1,y_1, LineWidth=2); plot(x_2,y_2, LineWidth=2);
hold off; axis([0,N,0,M]); title("angle = " + theta + " (degrees)")



% TODO: *** git ***


%%
function allevents(~, ~, ~) % dummy  to catch the trim rectangle move event
end
