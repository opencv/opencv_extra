dir_mat = 'VOC2010/';
dir_xml = 'models_VOC2010/';
mkdir(dir_xml);
fs = dir('VOC2010/*.mat');
for i = 1 : length(fs)
    fname = fs(i).name;
    if strcmp('person_grammar_final.mat', fname)
        continue;
    end
    fprintf('\n%s', fname);
    fname_in = [dir_mat fname];
    fname_out = [dir_xml fname(1:end-10) '.xml'];
    mat2xml_release5_LatentSVM(fname_in, fname_out);
end