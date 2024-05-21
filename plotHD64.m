%% plotHD64 - A function to shade in the contacts on the HD64 SVG
% Author: Sam Parker
% Last Modified: 13-May-2025

function svg = plotHD64(varargin)
    if nargin == 1
        electrode = varargin{1};
        if length(electrode) == 1
            color = ["#ff0000"]; 
        elseif length(electrode) == 2
            color = ["#0000ff"]; 
        else
            color = ["#33bbee", "#ee7733", "#ee3377", "#ffb508", "#8817b1", "009988"];
        end
    elseif nargin == 2
        if class(varargin{2}) == "double"
            electrode = [varargin{1}, varargin{2}];
            color = [repelem(["#ff0000"], length(varargin{1})), repelem(["#0000ff"], length(varargin{2}))];
        elseif class(varargin{2}) == "string"
            electrode = varargin{1};
            color = varargin{2};
        end
    end

    svg = fileread("HD64.svg");
    style_start_idx = strfind(svg, "}");
    style_start_idx = style_start_idx(1);
    header = svg(1:style_start_idx);
    footer = svg((style_start_idx+1):end);
    cur_style_idx = 1;
    custom_style_classes = "";
    unique_colors = unique(color);
    
    for i = 1:length(unique_colors)
        cur_style_idx = cur_style_idx+1;
        custom_style_classes = sprintf("%s\n      .cls-%d {\n        fill: %s;\n        stroke: #231f20;\n        stroke-miterlimit: 10;\n      }", custom_style_classes, cur_style_idx, unique_colors(i));
    end
    svg = char(strcat(header, custom_style_classes, footer));

    hd64_order = [60, 55, 58, 63, ...
        24, 54, 47, 46, 53, 52, 59, 25, ...
        23, 38, 21, 20, 29, 28, 45, 26, ...
        22, 31, 10, 2, 7, 19, 36, 27, ...
        32, 30, 0, 13, 16, 9, 37, 35, ...
        48, 41, 11, 3, 6, 18, 42, 51, ...
        49, 39, 1, 4, 5, 8, 44, 50, ...
        56, 40, 12, 14, 15, 17, 43, 57];

    for i = 1:length(electrode)
        if electrode(i) > 64
            electrode(i) = electrode(i) - 128;
        end
        color_recirc = 1 + mod(i-1, length(color));
        color_idx = find(unique_colors == color(color_recirc));
        rect_idxs = strfind(svg, "<rect class=");
        rect_start_idx = rect_idxs(hd64_order == electrode(i)) + 17;
        if ~isempty(rect_start_idx)
            header = svg(1:rect_start_idx-1);
            footer = svg((rect_start_idx+1):end);
            svg = char(sprintf("%s%d%s", header, color_idx+1, footer));
        end
    end 
end