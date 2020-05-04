""" 
0: wav (in [-1, 1])
1: fs
2: text (str)
3: speaker id
4: chapter id
5: utt id


Let's reserve 10% speakers for testing (?) for the remaining, reserve 10% utt of each speaker.
"""

import torch
from torchaudio.datasets import LIBRISPEECH
# import librosa


URL = "train-clean-100"
FOLDER_IN_ARCHIVE = "LibriSpeech"


class LibriSpeech4SpeakerRecognition(LIBRISPEECH):
    def __init__(
        self, root, project_fs, subset, wav_length=32000, url=URL, folder_in_archive=FOLDER_IN_ARCHIVE, download=False,
        train_speaker_ratio=0.9, train_utterance_ratio=0.9,
    ):
        """
        subset: "train", "test", "outside"
        """
        super().__init__(root, url=url, folder_in_archive=folder_in_archive, download=download)
        self._split(subset, train_speaker_ratio, train_utterance_ratio)
        self.project_fs = project_fs
        self.wav_length = wav_length

    def _split(self, subset, train_speaker_ratio, train_utterance_ratio):
        def _parse(name_string):
            speaker_id, chapter_id, utterance_id = name_string.split("-")
            return speaker_id, chapter_id, utterance_id 
        utt_per_speaker = {}
        # for (wav, fs, text, spk, chap, utt) in ds:
        for filename in self._walker:
            speaker_id, chapter_id, utterance_id = _parse(filename)
            if utt_per_speaker.get(speaker_id, None) is None:
                utt_per_speaker[speaker_id] = [utterance_id]
            else:
                utt_per_speaker[speaker_id].append(utterance_id)
        # 
        speakers = list(utt_per_speaker.keys())
        speakers.sort()
        num_train_speaker = int(len(speakers) * train_speaker_ratio)
        speakers = {
            "train": speakers[:num_train_speaker],
            "test": speakers[num_train_speaker:]
        }
        self.speakers = {
            "train": [int(s) for s in speakers["train"]],
            "test": [int(s) for s in speakers["test"]],
        }
        # 
        for spk in speakers["train"]:
            utt_per_speaker[spk].sort()
            num_train_utterance = int(len(utt_per_speaker[spk]) * train_utterance_ratio)
            utt_per_speaker[spk] = {
                "train": utt_per_speaker[spk][:num_train_utterance],
                "test": utt_per_speaker[spk][num_train_utterance:],
            }
        # 
        trn_walker = []
        test_walker = []            
        outsiders = []
        for filename in self._walker:
            speaker_id, chapter_id, utterance_id = _parse(filename)
            # speaker_id = int(speaker_id)
            # import pdb; pdb.set_trace()
            if speaker_id in speakers["train"]:
                if utterance_id in utt_per_speaker[speaker_id]["train"]:
                    trn_walker.append(filename)
                else:
                    test_walker.append(filename)
            else:
                outsiders.append(outsiders)

        if subset == "train":
            self._walker = trn_walker
        elif subset == "test":
            self._walker = test_walker
        else:
            self._walker = outsiders

    def __getitem__(self, n):
        waveform, sample_rate, _, speaker_id, _, _ = super().__getitem__(n)

        n_channel, duration = waveform.shape
        if duration > self.wav_length:
            i = torch.randint(0, duration - self.wav_length, []).long()
            waveform = waveform[:, i: i + self.wav_length]
        else:
            waveform = torch.cat(
                [
                    waveform,
                    torch.zeros(n_channel, self.wav_length - duration)
                ],
                1
            )


        # import pdb; pdb.set_trace()
        # waveform = librosa.core.resample(waveform, sample_rate, self.project_fs)
        return waveform, self.speakers["train"].index(speaker_id)

        # fileid = self._walker[n]
        # return load_librispeech_item(fileid, self._path, self._ext_audio, self._ext_txt)



# ds = LibriSpeech4SpeakerRecognition(root="/data/speech", project_fs=16000, url='train-clean-100', folder_in_archive='LibriSpeech', download=True)
# dso = torchaudio.datasets.LIBRISPEECH(root="/data/speech", url='train-clean-100', folder_in_archive='LibriSpeech', download=True)


# len(ds)
# len(dso)

"""
from dev.loaders import LibriSpeech4SpeakerRecognition
ds = LibriSpeech4SpeakerRecognition(root="/data/speech", project_fs=16000, wav_length=32000, url='train-clean-100', folder_in_archive='LibriSpeech', download=True)
ds[0][0].shape

dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, num_workers=8)

wav, spk = next(iter(dl))

"""
