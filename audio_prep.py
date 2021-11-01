import os
import librosa
import math
import json

data = {
    "mapping": [],
    "mfcc": [],
    "labels": []
}
SAMPLE_RATE = 22050
DURATION = 30
SAMPLE_PER_TRACK = SAMPLE_RATE * DURATION
num_segments = 10
num_samples_per_segment = int(SAMPLE_PER_TRACK / num_segments)
n_fft = 2048
hop_length = 512
n_mfcc = 13
expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

dataset_path = "AudioData/genres_original"
json_path = "data_10.json"
for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
    if dirpath is not dataset_path:
        dirpath_components = dirpath.split("\\")
        semantic_label = dirpath_components[-1]
        data["mapping"].append(semantic_label)
        print("\nProcessing {}".format(semantic_label))

        for f in filenames:
            # Load audio files
            file_path = os.path.join(dirpath, f)
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            # process segments extracting mfcc and storing data
            for s in range(num_segments):
                start_sample = num_samples_per_segment * s
                finish_sample = start_sample + num_samples_per_segment

                mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                            sr=sr,
                                            n_fft=n_fft,
                                            n_mfcc=n_mfcc,
                                            hop_length=hop_length)
                mfcc = mfcc.T
                # store mfcc for segment if it has the expected length
                if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(i - 1)
                    print("{}, segment:{}".format(file_path, s + 1))

with open(json_path, "w") as fp:
    json.dump(data, fp, indent=4)
