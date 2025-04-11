import os
import sys
import glob
import random
import urllib.request
import tarfile
import numpy as np
import tensorflow as tf

from uNAS.dataset import Dataset

AUTOTUNE = tf.data.AUTOTUNE

class SpeechCommandsDataset(Dataset):
    def __init__(self, data_dir="./speech_dataset", sample_rate=16000, 
                 num_mfcc=13, fix_seeds=False):
        if fix_seeds:
            np.random.seed(42)
            tf.random.set_seed(42)
            random.seed(42)

        self._data_dir = data_dir
        self._sample_rate = sample_rate
        self._num_mfcc = num_mfcc
        self._input_shape = (49, num_mfcc)
        self._target_classes = ['yes', 'no', 'up', 'down', 'left', 
                              'right', 'on', 'off', 'stop', 'go']
        self._class_names = np.array(sorted(self._target_classes))
        self._num_classes = len(self._class_names)

        self._download_and_extract_dataset()

        self.train_ds = self._load_dataset(split='train')
        self.val_ds = self._load_dataset(split='validation')
        self.test_ds = self._load_dataset(split='test')

    def _download_and_extract_dataset(self):
        if not tf.io.gfile.exists(self._data_dir):
            remote = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
            local = self._data_dir + ".tar.gz"

            def progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading Speech dataset %.1f%%' %
                                 (float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            urllib.request.urlretrieve(remote, local, progress)
            tgz = tarfile.open(local)

            members = [m for m in tgz.getmembers()
                       if len(m.name.split('/')) > 1 and m.name.split('/')[1] in self._target_classes]

            print(f"\nExtracting target classes: {', '.join(self._target_classes)}")
            for i, member in enumerate(members):
                tgz.extract(member, path=self._data_dir)
                if i % 100 == 0 or i == len(members) - 1:
                    progress_percent = (i + 1) / len(members) * 100
                    sys.stdout.write(f'\rExtracting files: {i+1}/{len(members)} ({progress_percent:.1f}%)')
                    sys.stdout.flush()

            print("\nExtraction complete!")
            tgz.close()

    def _decode_audio(self, file_path):
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
        audio = tf.squeeze(audio, axis=-1)
        
        current_length = tf.shape(audio)[0]
        
        def pad_audio():
            padding = tf.zeros([self._sample_rate - current_length], dtype=tf.float32)
            return tf.concat([audio, padding], axis=0)
        
        def trim_audio():
            return audio[:self._sample_rate]
        
        audio = tf.cond(
            current_length < self._sample_rate,
            pad_audio,
            lambda: trim_audio()
        )
        
        return tf.ensure_shape(audio, [self._sample_rate])
    
    def _get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        return tf.argmax(tf.cast(tf.equal(self._class_names, label), tf.int32))

    def _get_mfcc(self, audio):
        audio = tf.ensure_shape(audio, [self._sample_rate])
        
        # Calculate STFT with consistent parameters
        frame_length = 640  # 40ms at 16kHz
        frame_step = 320    # 20ms at 16kHz
        fft_length = 1024   # Number of FFT points
        
        # Compute STFT
        stft = tf.signal.stft(
            audio,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
            window_fn=tf.signal.hann_window,
            pad_end=True
        )
        spectrogram = tf.abs(stft)
        num_spectrogram_bins = fft_length // 2 + 1
        
        mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=60,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=self._sample_rate,
            lower_edge_hertz=20.0,
            upper_edge_hertz=4000.0
        )

        mel_spectrogram = tf.tensordot(spectrogram, mel_weight_matrix, 1)
        mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(mel_weight_matrix.shape[-1:]))
        log_mel = tf.math.log(mel_spectrogram + 1e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :self._num_mfcc]

        target_frames = 49
        mfcc_shape = tf.shape(mfccs)[0]

        def pad():
            padding = tf.zeros([target_frames - mfcc_shape, self._num_mfcc])
            return tf.concat([mfccs, padding], axis=0)

        def crop():
            return mfccs[:target_frames]

        mfccs = tf.cond(mfcc_shape < target_frames, pad, crop)
        mfccs = tf.ensure_shape(mfccs, [49, self._num_mfcc])
        return mfccs

    def _process_path(self, file_path):
        audio = self._decode_audio(file_path)
        mfcc = self._get_mfcc(audio)  # shape: [frames, num_mfcc]
        label = self._get_label(file_path)
        return mfcc, label
    
    def _load_example(self, file_path, label):
        """Load and process a single audio example."""
        if isinstance(file_path, tf.Tensor):
            file_path = tf.cast(file_path, tf.string)
        
        audio = self._decode_audio(file_path)
        mfcc = self._get_mfcc(audio)
        label = tf.cast(label, tf.int32)
        
        return mfcc, label

    def _split_files(self, split):
        """Split entire dataset into train/val/test."""
        all_files = []
        for class_name in self._target_classes:
            class_files = glob.glob(os.path.join(self._data_dir, class_name, '*.wav'))
            for f in class_files:
                all_files.append((f, class_name))

        random.shuffle(all_files)
        n = len(all_files)
        train_split = int(n * 0.8)
        val_split = int(n * 0.9)

        if split == 'train':
            subset = all_files[:train_split]
        elif split == 'validation':
            subset = all_files[train_split:val_split]
        else:
            subset = all_files[val_split:]

        file_paths = [f for f, _ in subset]
        labels = [self._class_names.tolist().index(c) for _, c in subset]

        ds = tf.data.Dataset.from_tensor_slices((
            tf.constant(file_paths, dtype=tf.string),
            tf.constant(labels, dtype=tf.int32)
        ))
        return ds

    def _load_dataset(self, split):
        ds = self._split_files(split)
        ds = ds.map(self._load_example, num_parallel_calls=AUTOTUNE)
        return ds

    def train_dataset(self):
        return self.train_ds

    def validation_dataset(self):
        return self.val_ds

    def test_dataset(self):
        return self.test_ds

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def input_shape(self):
        return self._input_shape
