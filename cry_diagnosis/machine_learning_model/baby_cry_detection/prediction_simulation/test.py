import argparse
import logging
import os
import pickle
import timeit
import warnings
import librosa
import numpy as np
import noisereduce as nr
from baby_cry_detection.rpi_methods import Reader
from baby_cry_detection.rpi_methods.feature_engineer import FeatureEngineer
from baby_cry_detection.rpi_methods.majority_voter import MajorityVoter
from baby_cry_detection.rpi_methods.baby_cry_predictor import BabyCryPredictor

def split_audio(audio, sr, window_size, hop_length):
    num_segments = (len(audio) - window_size) // hop_length + 1
    return [audio[i*hop_length:i*hop_length+window_size] for i in range(num_segments)]

def extract_features(audio_segment, sr, n_mfcc):
    audio_segment = nr.reduce_noise(y=audio_segment, sr=sr)
    audio_segment = librosa.util.normalize(audio_segment)
    mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=n_mfcc)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path_data',
                        default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--load_path_model',
                        default='{}/../../../baby_cry_detection-master/output/model/'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--save_path',
                        default='{}/../../../baby_cry_detection-master/output/prediction/'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--file_name', default='audio_20240727_134244.wav')
    parser.add_argument('--log_path',
                        default='{}/../../'.format(os.path.dirname(os.path.abspath(__file__))))

    # Arguments
    args = parser.parse_args()
    load_path_data = os.path.normpath(args.load_path_data)
    load_path_model = os.path.normpath(args.load_path_model)
    file_name = args.file_name
    save_path = os.path.normpath(args.save_path)
    log_path = os.path.normpath(args.log_path)

    # Set up logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %I:%M:%S %p',
                        filename=os.path.join(log_path, 'logs_prediction_test_test_model.log'),
                        filemode='w',
                        level=logging.INFO)

    # READ RAW SIGNAL
    logging.info('Reading {0}'.format(file_name))
    start = timeit.default_timer()

    # Read signal
    file_path = os.path.join(load_path_data, file_name)
    audio_data, sample_rate = librosa.load(file_path, sr=None)

    stop = timeit.default_timer()
    logging.info('Time taken for reading file: {0}'.format(stop - start))

    # FEATURE ENGINEERING
    logging.info('Starting feature engineering')
    start = timeit.default_timer()

    # Split audio into 1-second windows with 0.5-second overlap
    window_size = sample_rate  # 1 second
    hop_length = sample_rate // 2  # 0.5 second
    audio_segments = split_audio(audio_data, sample_rate, window_size, hop_length)

    # Extract features from each segment
    n_mfcc = 18  # מספר התכונות הנכון שהמודל מצפה לקבל
    features_list = [extract_features(segment, sample_rate, n_mfcc) for segment in audio_segments]

    stop = timeit.default_timer()
    logging.info('Time taken for feature engineering: {0}'.format(stop - start))

    # MAKE PREDICTION
    logging.info('Predicting...')
    start = timeit.default_timer()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        with open(os.path.join(load_path_model, 'model.pkl'), 'rb') as fp:
            model = pickle.load(fp)

    predictions = []
    for features in features_list:
        features = np.expand_dims(features, axis=0)  # Expand dimensions to match model input
        prediction = model.predict(features)
        predicted_label = np.argmax(prediction, axis=1)[0]
        predictions.append(predicted_label)

    # MAJORITY VOTE
    majority_voter = MajorityVoter(predictions)
    majority_vote = majority_voter.vote()

    stop = timeit.default_timer()
    logging.info('Time taken for prediction: {0}. Is it a baby cry?? {1}'.format(stop - start, majority_vote))

    # SAVE
    logging.info('Saving prediction...')
    with open(os.path.join(save_path, 'prediction.txt'), 'w') as text_file:
        text_file.write("{}".format(majority_vote))

    logging.info('Saved! {}'.format(os.path.join(save_path, 'prediction.txt')))

if __name__ == '__main__':
    main()
