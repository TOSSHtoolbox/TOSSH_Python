function generate_reference(tossh_path)
%generate_reference runs the MATLAB TOSSH functions listed in cases.json
%   and writes their outputs to reference/<case name>.mat.
%
%   INPUT
%   tossh_path: path to the MATLAB TOSSH repository
%
%   Usually called via generate_reference.py, but can also be run directly
%   in MATLAB, e.g. generate_reference('C:/Projects/_TOSSH/TOSSH')

here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(tossh_path, 'TOSSH_code')));

% load example data (same file as used by the Python tests)
data_file = fullfile(here, '..', '..', 'example', 'example_data', '33029_daily.csv');
opts = detectImportOptions(data_file);
opts = setvartype(opts, 't', 'datetime');
opts = setvaropts(opts, 't', 'InputFormat', 'dd-MMM-yyyy', 'DatetimeLocale', 'en_US');
data = readtable(data_file, opts);

% load cases (jsondecode returns a struct array if all cases have the same
% structure, otherwise a cell array)
cases = jsondecode(fileread(fullfile(here, 'cases.json')));
if isstruct(cases)
    cases = num2cell(cases);
end

out_dir = fullfile(here, 'reference');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

nr_updated = 0;
for i = 1:numel(cases)
    c = cases{i};
    fprintf('Running %s\n', c.name);

    % collect input time series
    inputs = cell(1, numel(c.inputs));
    for k = 1:numel(c.inputs)
        if ischar(c.inputs{k})
            inputs{k} = data.(c.inputs{k}); % column of the example data
        else
            inputs{k} = c.inputs{k}; % fixed value, e.g. x of sig_x_percentile
        end
    end

    % convert options to name-value pairs
    names = fieldnames(c.options);
    name_value = {};
    for k = 1:numel(names)
        name_value = [name_value, {names{k}, c.options.(names{k})}];
    end

    % call function and keep all outputs
    out = cell(1, nargout(c.function));
    [out{:}] = feval(c.function, inputs{:}, name_value{:});

    % outputs that are not compared (e.g. figure handles) are stored as empty
    for k = 1:numel(out)
        if ~(isnumeric(out{k}) || islogical(out{k}) || ischar(out{k}))
            out{k} = [];
        end
    end

    % one file per case, readable in Python with scipy.io.loadmat. The file
    % is only written if the values changed, as the file header contains a
    % timestamp and git would otherwise show unchanged files as modified.
    outputs = out;
    file = fullfile(out_dir, [c.name '.mat']);
    if exist(file, 'file')
        old = load(file, 'outputs');
        if isequaln(old.outputs, outputs)
            continue
        end
    end
    save(file, 'outputs', '-v7');
    nr_updated = nr_updated + 1;
    fprintf('  -> %s.mat written (new or changed values)\n', c.name);
end

if nr_updated == 0
    fprintf('No reference values changed.\n');
    return
end

% store info on where the reference values come from
[~, commit] = system(sprintf('git -C "%s" rev-parse HEAD', tossh_path));
info = struct( ...
    'matlab_version', version, ...
    'tossh_commit', strtrim(commit), ...
    'created', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm')));
fid = fopen(fullfile(out_dir, 'info.json'), 'w');
fprintf(fid, '%s', jsonencode(info, 'PrettyPrint', true));
fclose(fid);
fprintf('%d reference file(s) updated in %s\n', nr_updated, out_dir);

end
