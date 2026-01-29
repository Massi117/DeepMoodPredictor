function plot_ctx_and_subctx(data, h, rois, cmap_type,to_flip,limits,oldannot,inflated_surf, title, colorbar_legend, ...
    axes_to_plot, scaling_factor)

% ---------- REQUIREMENTS:
% https://www.mathworks.com/matlabcentral/fileexchange/120088-200-colormap 
% https://enigma-toolbox.readthedocs.io/en/latest/index.html
% freesurfer must be installed and its path must be provided to matlab

% ---------- INPUTS:
% data (size Nx1, numeric): values of what you want to plot (e.g. correlation values, t-values, etc)

% h (size Nx1, boolean): whether a value is significant. Non-significant values are plotted in grayscale

% rois (size Nx1, cell array of strings): region name of each data point
    %   must include the following subcortical regions: {'Accumbens-area', 'Amygdala', 'Caudate', 'Hippocampus', ...
    %       'Pallidum' 'Putamen', 'Thalamus-Proper'}
    %   any included ctx areas must match FreeSurfer .annot name conventions (so it should not include 'ctx-lh' or 'ctx-rh' prefixes)
    %   you can include as many ctx areas as you want; missing areas will be shown in black

% cmap_type: colormap name, or 'redblue' (makes a diverging colormap with white=0, positive=red, negative=blue)
    % see here for colormap names: https://www.mathworks.com/matlabcentral/fileexchange/120088-200-colormap

% to_flip: whether to flip cmap colors (0 or 1)
% limits: numeric array of size 2x1 if you want to manually set color limits, otherwise empty variable
% oldannot: location of lh.aparc.annot file
% inflated_surf: location of lh.inflated file
% title: char (or empty for no title)
% colorbar_legend: char (or empty for no title)
% axes_to_plot: handle of axes (e.g. from a subplot or tiledlayout) or empty to create new ones
% scaling_factor: a numeric value to change the size of the surfaces (below 1 to shrink, above 1 to enlargen)

% ---------- EXAMPLE USAGE:
% plot_ctx_and_subctx(data_vector, h_vector, roi_names, 'YlOrRd', 0, [-1 1], ...
%               '/location/of/file/lh.aparc.annot','/location/of/file/lh.inflated', ...
%             'EEG correlated with BOLD', 'correlation (r)', [])

% ---------- EXAMPLE USAGE WITH AXES TARGETING:
% KNOWN ISSUE: when using tiledlayout (and possibly subplots) in 2021b (and possibly other versions), the first column
% of the figure layout ends up offset to the right. A workaround is to skip plotting in the first column. 
%
% tiledlayout('TileSpacing','none', 'Padding','tight')
% nexttile
% tileAx = gca;
% plot_ctx_and_subctx(data_vector, h_vector, roi_names, 'YlOrRd', 0, [-1 1], ...
%               '/location/of/file/lh.aparc.annot','/location/of/file/lh.inflated', ...
%             'EEG correlated with BOLD', 'correlation (r)', tileAx)

[surf_vertices, surf_faces] = freesurfer_read_surf(inflated_surf);

expected_sbctx_order = {'Accumbens-area', 'Amygdala', 'Caudate', 'Hippocampus', ...
    'Pallidum' 'Putamen', 'Thalamus-Proper'};
[~, sbctx_rois_idx] = ismember(expected_sbctx_order, rois);

ctx_rois_idx = cellfun(@(str) isstrprop(str(1), 'lower'), rois);

if length(limits)==2
    min_val = limits(1);
    max_val = limits(2);
else
    if strcmp(cmap_type,'redblue')
        abs_max = max(abs([min(data(:)), max(data(:))]));
        min_val = -abs_max;
        max_val = abs_max;
    else
        min_val = min(data(:));
        max_val = max(data(:));
    end
end

[colors, ~, data_colors] = create_color_map(data, min_val, max_val, 'cmap_type', cmap_type, 'to_flip', to_flip);
[~, ~, data_colors_notsig] = create_color_map(data, min_val, max_val, 'cmap_type', 'Greys', 'to_flip', to_flip);

if strcmp(cmap_type, 'redblue')
data_colors_notsig = repmat([.4 .4 .4], size(data_colors,1),1);
end

data_colors(h==0,:) = data_colors_notsig(h==0,:);

[~, label_names, colortable] = annot_from_colorbar(oldannot, rois(ctx_rois_idx), data_colors(ctx_rois_idx,:), [], []);

[~, idx_vtx] = ismember(label_names, colortable.table(:,5));
face_colors = colortable.table(idx_vtx,1:3);
face_colors = face_colors./255;

if ~isempty(axes_to_plot)
    refPos = axes_to_plot.Position; % [x y w h] in normalized figure or parent units
else
    refPos = [0 0 1 1]; % default to whole figure if not specified
end

relPos = [0, 0, 0.75*scaling_factor, 0.75*scaling_factor];
axes('OuterPosition', rel2absPos(refPos, relPos));
plot_surface(surf_vertices, surf_faces, face_colors, [90 0])

relPos = [0.35, 0.22, 0.75*scaling_factor, 0.75*scaling_factor];
axes('OuterPosition', rel2absPos(refPos, relPos));
plot_surface(surf_vertices, surf_faces, face_colors, [-90 0])

relPos = [0, 0.4, 0.6*scaling_factor, 0.6*scaling_factor];
axes('OuterPosition', rel2absPos(refPos, relPos));
plot_subcortical_custom([data_colors(sbctx_rois_idx,:); data_colors(sbctx_rois_idx,:)], 'ventricles', 'False');

colormap(colors); 
cb = colorbar; axis off
caxis([min_val, max_val]);
set(cb, 'Orientation', 'horizontal');
set(cb, 'Position', rel2absPos(refPos, [0.55 0.20 0.3 0.03]));
set(cb, 'XAxisLocation', 'bottom');
h = get(cb, 'Title');
set(h, 'String', colorbar_legend);
% sctx.CDataMapping = 'direct';

if ~isempty(title)
annotation('textbox', rel2absPos(refPos, [0, 0.82, 1, 0.05]), ... % [x y w h] in normalized figure units
    'String', title,...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'LineStyle', 'none', ...
    'FontSize', 14, ...
    'FontWeight', 'bold',...
    'Tag','custom_title');
end

if ~isempty(axes_to_plot)
    delete(axes_to_plot);
end

end

function [colors, color_idx, data_color] = create_color_map(data, varargin)

default_min_value = min(data(:));
default_max_value = max(data(:));

p = inputParser;
addRequired(p,'data', @(x) isnumeric(x));
addOptional(p,'min_value',default_min_value, @(x) isnumeric(x) && isscalar(x));
addOptional(p,'max_value',default_max_value, @(x) isnumeric(x) && isscalar(x));
addParameter(p,'cmap_type','cool');
addParameter(p,'to_flip',1);
parse(p,data,varargin{:});

min_value = p.Results.min_value;
max_value = p.Results.max_value;
cmap_type = p.Results.cmap_type;
to_flip = p.Results.to_flip;

% colors: full colormap
% colors_idx: used for indexing each value from data from the color map
% data_color: color info for each point of data

if ~strcmp(cmap_type, 'redblue')

    indices = 1:256;

        cmap = slanCM(cmap_type);
%         cmap = colormap(cmap_type);

    xref = linspace(min_value,max_value,length(indices));
    colors = cmap(indices,:);

    if to_flip==1
        colors = flip(colors);
    end

elseif strcmp(cmap_type, 'redblue')

    indices = 1:255;

    n = length(indices);

    abs_max = max(abs(min_value), abs(max_value));
    xref = linspace(-abs_max,abs_max,n);

    step_size = 1/(floor(n/2));

    B = [ones(1,floor(n/2)), 1:-step_size:0];
    G = [0:step_size:1, 1-step_size:-step_size:0];
    R = [0:step_size:1, ones(1,floor(n/2))];
    
    colors = [R; G; B]';
end

if size(data,1)==1 || size(data,2)==1
    color_idx = interp1(xref,1:length(xref),data,'nearest');
    data_color = colors(color_idx,:);
else
    color_idx = interp1(xref,1:length(xref),data,'nearest');
    data_color = nan([size(data,1) 3 size(data,2)]);
    for i=1:size(data,2)        
        data_color(:,:,i) = colors(color_idx(:,i),:);
    end
end

end

function [v, newl, ctnew] = annot_from_colorbar(oldannot, rois_hdr, colors, folder_to_save, newannot)

% rois_hdr: roi names that match freesurfer labels
% data: matching data


% % edit roi names so it matches freesurfer labels
% for i=1:length(rois_hdr)
%     rois_hdr{i}=rois_hdr{i}(8:end);
% end

%% load old lookup table
[v,l,ct]=read_annotation(oldannot);

% color table roi names
ct_hdr=ct.struct_names;

% color table colors
color_table = ct.table;

% find missing rois
[~, loc_missing]= ismember(ct_hdr, rois_hdr);

% set color of missing rois to black
color_table(loc_missing==0,1:3) = zeros(sum(loc_missing==0),3);

% get indexes of rois that we do have in original roi list
[~, loc]= ismember(rois_hdr, ct_hdr);

% set color of those rois to desired colors
color_table(loc,1:3) = round(colors*255); % freesurfer uses 255 range
color_table(:,5) = color_table(:,1) + color_table(:,2)*2^8 + color_table(:,3)*2^16; % labels

% make new table
ctnew=ct;
ctnew.table=color_table;

%% match new label values with label values in old label file

old_labels = ct.table(:,5);

% find correspondence between labels and colors
[~, loc]=ismember(l,old_labels);

% to avoid indexing to 0 (for missing labels)
loc(loc==0) = length(color_table)+1;
color_table(end+1,:) = [0 0 0 0 0];

% make new label files with new colors
newl = color_table(loc,5);

%%
if ~isempty(folder_to_save)
    cd(folder_to_save)

    write_annotation(newannot,v,newl,ctnew);
end
end

function h = plot_subcortical_custom(data, varargin)
%
% Usage:
%   [a, cb] = plot_subcortical(data, varargin);   
%
% Description:
%   Plot subcortical surface with lateral and medial views (authors: @saratheriver)
%
% Inputs:
%   data (double array) - vector of data, size = [1 x v]. One value per 
%       subcortical structure, in this order: L-accumbens, L-amygdala, 
%       L-caudate, L-hippocampus, L-pallidum L-putamen, L-thalamus, 
%       L-ventricle, R-accumbens, R-amygdala, R-caudate, R-hippocampus, 
%       R-pallidum, R-putamen, R-thalamus, R-ventricle
%
% ALTERNATIVE: if you want to directly set the color of each region, data can be v x 3
%
% Name/value pairs:
%   ventricles (string, optional) - If 'True' (default) shows the ventricles 
%       (data must be size = [1 x 16]). If 'False', then ventricles are not 
%       shown and data must be size = [1 x 14].
%   label_text (string, optional) - Label text for colorbar. Default is empty.
%   background (string, double array, optional) - Background color. 
%       Default is 'white'.
%   color_range (double array, optional) - Range of colorbar. Default is 
%       [min(data) max(data)].
%   cmap (string, double array, optional) - Colormap name. Default is 'RdBu_r'.
%
% Outputs:
%   a (axes) - vector of handles to the axes, left to right, top to bottom
%   cb (colorbar) - colorbar handle
%
% Sara Lariviere  |  saratheriver@gmail.com
% modified by ljacob@mit.edu

p = inputParser;
addParameter(p, 'ventricles', 'True', @ischar);
addParameter(p, 'label_text', "", @ischar);
addParameter(p, 'background', 'white', @ischar);
addParameter(p, 'color_range', [min(data) max(data)], @isnumeric);
addParameter(p, 'cmap', 'RdBu_r', @ischar);

% Parse the input
parse(p, varargin{:});
in = p.Results;

% load subcortical templates
surf_lh = SurfStatReadSurf('sctx_lh');
%surf_rh = SurfStatReadSurf('sctx_rh');

% super inefficient way to attribute vertices to subcortical areas
if size(data,1)==1
    data_colors = [];
    if strcmp(in.ventricles, 'True')
        data = [repmat(data(1), 867, 1); repmat(data(2), 1419, 1); ...
                repmat(data(3), 3012, 1); repmat(data(4), 3784, 1); ...
                repmat(data(5), 1446, 1); repmat(data(6), 4003, 1); ...
                repmat(data(7), 3726, 1); repmat(data(8), 7653, 1);...
                repmat(data(9), 838, 1); repmat(data(10), 1457, 1); ...
                repmat(data(11), 3208, 1); repmat(data(12), 3742, 1); ...
                repmat(data(13), 1373, 1); repmat(data(14), 3871, 1); ...
                repmat(data(15), 3699, 1); repmat(data(16), 7180, 1)];
    elseif strcmp(in.ventricles, 'False')
        data1 = nan(16, 1);
        data1([1:7 9:15]) = data;
        
        data = [repmat(data1(1), 867, 1); repmat(data1(2), 1419, 1); ...
                repmat(data1(3), 3012, 1); repmat(data1(4), 3784, 1); ...
                repmat(data1(5), 1446, 1); repmat(data1(6), 4003, 1); ...
                repmat(data1(7), 3726, 1); repmat(data1(8), 7653, 1);...
                repmat(data1(9), 838, 1); repmat(data1(10), 1457, 1); ...
                repmat(data1(11), 3208, 1); repmat(data1(12), 3742, 1); ...
                repmat(data1(13), 1373, 1); repmat(data1(14), 3871, 1); ...
                repmat(data1(15), 3699, 1); repmat(data1(16), 7180, 1)];
        
        data([18258:25910 44099:end]) = nan;
    end
else % support for ventricles missing, can add later
    data1 = nan(16, 3);
    data1([1:7 9:15],:) = data;
    
    data_colors = [repmat(data1(1,:), 867, 1); repmat(data1(2,:), 1419, 1); ...
            repmat(data1(3,:), 3012, 1); repmat(data1(4,:), 3784, 1); ...
            repmat(data1(5,:), 1446, 1); repmat(data1(6,:), 4003, 1); ...
            repmat(data1(7,:), 3726, 1); repmat(data1(8,:), 7653, 1);...
            repmat(data1(9,:), 838, 1); repmat(data1(10,:), 1457, 1); ...
            repmat(data1(11,:), 3208, 1); repmat(data1(12,:), 3742, 1); ...
            repmat(data1(13,:), 1373, 1); repmat(data1(14,:), 3871, 1); ...
            repmat(data1(15,:), 3699, 1); repmat(data1(16,:), 7180, 1)];
    
    data_colors([18258:25910 44099:end],:) = nan;
    data = data_colors(:,1);
end
            
vl   = 1:size(surf_lh.coord, 2);

h = trisurf(surf_lh.tri,surf_lh.coord(1,:),surf_lh.coord(2,:),surf_lh.coord(3,:),...
    double(data(vl)),'EdgeColor','none');
view(90,0); 
daspect([1 1 1]); axis tight; camlight; axis vis3d off;
lighting phong; material dull; shading flat;

if ~isempty(data_colors)
h.CDataMapping = 'direct';
h.FaceVertexCData = data_colors(vl,:);
end

return
end

function absPos = rel2absPos(refPos, relPos)
% rel2absPos Convert relative [x y w h] to absolute [x y w h] within a reference rectangle.
%
%   absPos = rel2absPos(refPos, relPos)
%
%   refPos: [x0 y0 w0 h0] reference position (e.g., axes.Position)
%   relPos: [xr yr wr hr] relative position (normalized to refPos)
%   absPos: absolute position within the figure or container

    absPos = [ ...
        refPos(1) + relPos(1)*refPos(3), ...
        refPos(2) + relPos(2)*refPos(4), ...
        relPos(3)*refPos(3), ...
        relPos(4)*refPos(4)];
end