% List of input EEG CSV files
file_list = {
    'EEG_Only_Subject1_Updated.csv',
    'EEG_Only_Subject2_Updated.csv',
    'EEG_Only_Subject3_Updated.csv',
    'EEG_Only_Subject4_Updated.csv',
    'EEG_Only_Subject5_Updated.csv',
    'EEG_Only_Subject6_Updated.csv'
};

% Sampling rate (CytonDaisy board)
Fs = 125;

for fileIdx = 1:length(file_list)
    filename = file_list{fileIdx};
    fprintf("\nProcessing: %s\n", filename);

    %% Step 1: Load the EEG Data
    data = readtable(filename);  % Load EEG table

    % Extract raw EEG signals (Channels are in columns 1 to 16)
    raw_eeg = data{:, 1:16};  % Get matrix form
    labels = data.Activity_Label_OpenBCI;  % Extract activity labels

    % Preview
    disp("Raw EEG dimensions:");
    disp(size(raw_eeg));

    %% Step 2: Check and Keep Channels with >50% Usable Data
    usable_threshold = 0.5;
    good_channels = [];

    for ch = 1:size(raw_eeg, 2)
        signal = raw_eeg(:, ch);
        % If more than 50% of the data is finite (i.e., not NaN or Inf)
        if sum(isfinite(signal)) / length(signal) >= usable_threshold
            good_channels(end+1) = ch;  % Keep this channel
        end
    end

    % Filter raw EEG data to only include good channels
    eeg_data = raw_eeg(:, good_channels);
    numChannels = length(good_channels);

    fprintf("Using %d good channels out of 16.\n", numChannels);

    %% Step 3: Filter the EEG (Bandpass 1–45 Hz)
    [b, a] = butter(4, [1 45] / (Fs / 2), 'bandpass');
    eeg_data_filtered = filtfilt(b, a, eeg_data);

    %% Step 4: Sliding Window Parameters
    windowSize = Fs * 2;                % 2 seconds = 250 samples
    stepSize = round(windowSize * 0.5); % 50% overlap = 125 samples

    features_all = [];
    labels_all = [];

    %% Step 5: Feature Extraction
    for startIdx = 1:stepSize:(size(eeg_data_filtered, 1) - windowSize)
        segment = eeg_data_filtered(startIdx:startIdx+windowSize-1, :);
        label_window = labels(startIdx:startIdx+windowSize-1);
        majority_label = mode(label_window);

        feature_vector = [];

        for ch = 1:numChannels
            signal = segment(:, ch);

            % -- Time-Domain Features (Only STD and RMS kept) --
            std_val = std(signal);
            rms_val = rms(signal);
            time_feats = [std_val, rms_val];

            % -- Frequency-Domain Features --
            [pxx, f] = pwelch(signal, [], [], [], Fs);
            delta = bandpower(pxx, f, [0.5 4], 'psd');
            theta = bandpower(pxx, f, [4 8], 'psd');
            alpha = bandpower(pxx, f, [8 13], 'psd');
            beta  = bandpower(pxx, f, [13 30], 'psd');
            gamma = bandpower(pxx, f, [30 45], 'psd');
            freq_feats = [delta, theta, alpha, beta, gamma];

            feature_vector = [feature_vector, time_feats, freq_feats];
        end

        features_all = [features_all; feature_vector];
        labels_all = [labels_all; majority_label];
    end

    %% Step 6: Create Table & Save
    num_features = size(features_all, 2);
    feature_names = strings(1, num_features);

    for i = 1:numChannels
        base = sprintf('Ch%d_', good_channels(i));  % Using actual channel number
        idx = (i-1)*7;
        feature_names(idx+1:idx+2) = base + ["Std", "RMS"];
        feature_names(idx+3:idx+7) = base + ["Delta", "Theta", "Alpha", "Beta", "Gamma"];
    end

    T = array2table(features_all, 'VariableNames', cellstr(feature_names));
    T.Activity_Label_OpenBCI = labels_all;

    % Construct correct output filename: EEG_Features_Only_S{n}-ValidChannels.csv
    subject_number = regexp(filename, 'Subject(\d+)', 'tokens');
    subject_id = subject_number{1}{1};  % Extract number as string
    output_filename = sprintf('EEG_Features_Only_S%s-ValidChannels.csv', subject_id);
    writetable(T, output_filename);

    fprintf("Features saved to %s\n", output_filename);
end
