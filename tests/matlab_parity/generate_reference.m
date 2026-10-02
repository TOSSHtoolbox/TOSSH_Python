function generate_reference(tossh_path)
%generate_reference runs the MATLAB TOSSH functions listed in cases.json
%   and writes their outputs to reference/reference.json.
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

results = struct();
for i = 1:numel(cases)
    c = cases{i};
    fprintf('Running %s\n', c.name);

    % collect input time series
    inputs = cell(1, numel(c.inputs));
    for k = 1:numel(c.inputs)
        inputs{k} = data.(c.inputs{k});
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

    % outputs that cannot be stored in JSON (e.g. figure handles) are skipped
    for k = 1:numel(out)
        if ~(isnumeric(out{k}) || islogical(out{k}) || ischar(out{k}))
            out{k} = '<not compared>';
        end
    end
    results.(c.name) = out;
end

% store info on where the reference values come from
[~, commit] = system(sprintf('git -C "%s" rev-parse HEAD', tossh_path));
reference.info = struct( ...
    'matlab_version', version, ...
    'tossh_commit', strtrim(commit), ...
    'created', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm')));
reference.results = results;

out_dir = fullfile(here, 'reference');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
fid = fopen(fullfile(out_dir, 'reference.json'), 'w');
fprintf(fid, '%s', jsonencode(reference, 'PrettyPrint', true));
fclose(fid);
fprintf('Reference values written to %s\n', fullfile(out_dir, 'reference.json'));

end
