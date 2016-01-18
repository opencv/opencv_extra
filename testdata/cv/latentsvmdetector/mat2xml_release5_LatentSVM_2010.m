% https://github.com/amc-nu/Latent-SVM-MAT2XML-Converter

function [] = mat2xml_release5_LatentSVM_2010(fname_in, fname_out)
load(fname_in);

num_feat = 31;

layers = 1;
components = 1:2:length(model.rules{model.start});

%% Write XML Header
f = fopen(fname_out, 'wb');
fprintf(f, '<Model>\n');
fprintf(f, '\t<!-- Number of components -->\n');
fprintf(f, '\t<NumComponents>%d</NumComponents>\n', length(components));
fprintf(f, '\t<!-- Number of features -->\n');
fprintf(f, '\t<P>%d</P>\n', num_feat);
fprintf(f, '\t<!-- Score threshold -->\n');
fprintf(f, '\t<ScoreThreshold>%.16f</ScoreThreshold>\n', model.thresh);

k = 1;
for c = components
  for layer = layers
    rhs = model.rules{model.start}(c).rhs;
    root = -1;
    parts = [];
    defs = {};
    anchors = {};
    
    if model.symbols(rhs(1)).type == 'T'
      root = model.symbols(rhs(1)).filter;
    else
      root = model.symbols(model.rules{rhs(1)}(layer).rhs).filter;
    end
    for i = 2:length(rhs)
      defs{end+1} = model_get_block(model, model.rules{rhs(i)}(layer).def);
      anchors{end+1} = model.rules{model.start}(c).anchor{i};
      fi = model.symbols(model.rules{rhs(i)}(layer).rhs).filter;
      parts = [parts fi];
    end
    
    numparts = length(parts);
    rootfilter = model_get_block(model, model.filters(root));
    %% Write Root filter
    fprintf(f, '\t<Component>\n');
    fprintf(f, '\t\t<!-- Root filter description -->\n');
    fprintf(f, '\t\t<RootFilter>\n');
    fprintf(f, '\t\t\t<!-- Dimensions -->\n');
    fprintf(f, '\t\t\t<sizeX>%d</sizeX>\n', size(rootfilter,2) );
    fprintf(f, '\t\t\t<sizeY>%d</sizeY>\n', size(rootfilter,1) );
    fprintf(f, '\t\t\t<!-- Weights (binary representation) -->\n');
    fprintf(f, '\t\t\t<Weights>');
    for jj = 1:size(rootfilter,1)
        for ii = 1:size(rootfilter,2)
            for kk = 1:num_feat
                fwrite(f, rootfilter(jj, ii, kk), 'double');
            end
        end
    end
    fprintf(f, '\t\t\t</Weights>\n');
    fprintf(f, '\t\t\t<!-- Linear term in score function -->\n');
    bias = model_get_block(model, model.rules{model.start}(c).offset) * model.features.bias;
    fprintf(f, '\t\t\t<LinearTerm>%.16f</LinearTerm>\n', bias);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf(f, '\t\t</RootFilter>\n\n');
    
    %% Write Part Filters
    fprintf(f, '\t\t<!-- Part filters description -->\n');
    fprintf(f, '\t\t<PartFilters>\n');
    fprintf(f, '\t\t\t<NumPartFilters>%d</NumPartFilters>\n', numparts);
    for j = 1:numparts
        partfilter = model_get_block(model, model.filters(parts(j)));%%%%%%%%%%%%%%%%%%%%%%%%
        fprintf(f, '\t\t\t<!-- Part filter %d description -->\n', j);
        fprintf(f, '\t\t\t<PartFilter>\n');
        fprintf(f, '\t\t\t\t<!-- Dimensions -->\n');
        fprintf(f, '\t\t\t\t<sizeX>%d</sizeX>\n', size(partfilter,2));
        fprintf(f, '\t\t\t\t<sizeY>%d</sizeY>\n', size(partfilter,1));
        fprintf(f, '\t\t\t\t<!-- Weights (binary representation) -->\n');
        fprintf(f, '\t\t\t\t<Weights>');
        for jj = 1:size(partfilter,1)
            for ii = 1:size(partfilter,2)
                for kk = 1:num_feat
                    fwrite(f, partfilter(jj, ii, kk), 'double');
                end
            end
        end
        fprintf(f, '\t\t\t\t</Weights>\n');
        fprintf(f, '\t\t\t\t<!-- Part filter offset -->\n');
        fprintf(f, '\t\t\t\t<V>\n');
        fprintf(f, '\t\t\t\t\t<Vx>%d</Vx>\n', anchors{j}(1));
        fprintf(f, '\t\t\t\t\t<Vy>%d</Vy>\n', anchors{j}(2));
        fprintf(f, '\t\t\t\t</V>\n');
        fprintf(f, '\t\t\t\t<!-- Quadratic penalty function coefficients -->\n');
        fprintf(f, '\t\t\t\t<Penalty>\n');
        fprintf(f, '\t\t\t\t\t<dx>%.16f</dx>\n', defs{j}(2));
        fprintf(f, '\t\t\t\t\t<dy>%.16f</dy>\n', defs{j}(4));
        fprintf(f, '\t\t\t\t\t<dxx>%.16f</dxx>\n', defs{j}(1));
        fprintf(f, '\t\t\t\t\t<dyy>%.16f</dyy>\n', defs{j}(3));
        fprintf(f, '\t\t\t\t</Penalty>\n');
         fprintf(f, '\t\t\t</PartFilter>\n');
        
    end
    fprintf(f, '\t\t</PartFilters>\n');
    fprintf(f, '\t</Component>\n');
    k = k+1;
  end
end
fprintf(f, '</Model>');
fclose(f);

%% Function modified from latent-svm-release-5
function w = model_get_block(m, obj)
bl    = obj.blocklabel;
shape = m.blocks(bl).shape;
type  = m.blocks(bl).type;
w     = reshape(m.blocks(bl).w, shape);

% Flip (if needed) according to block type
switch(type)
  case 'F'
    if obj.flip
      w = flipfeat(w);
    end
  case 'P'
    if obj.flip
      w = reshape(m.blocks(bl).w_flipped, shape);
    end
  case 'D'
    if obj.flip
      w(2) = -w(2);
    end
  %case block_types.Other % 'O'
end