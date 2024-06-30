import os
import numpy as np
# -------------#
import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
# -------------#
from .task import generate_masked_acoustic_model_data
from pretrain.bucket_dataset import FeatDataset

HALF_BATCHSIZE_TIME = 99999


# 直接给定了feature
class KaldiAcousticDataset(FeatDataset):

    def __init__(self, extracter, task_config, bucket_size, file_path, sets,
                 max_timestep=0, libri_root=None, **kwargs):
        super(KaldiAcousticDataset, self).__init__(extracter, task_config, bucket_size, file_path, sets,
                                                   max_timestep, libri_root, **kwargs)

    # 加载feature
    def _load_feat(self, feat_path):
        if self.libri_root is None:
            return torch.FloatTensor(np.load(os.path.join(self.root, feat_path)))
        else:
            # torchaudio加载声音文件
            # wav: (1, seq_len)
            wav, _ = torchaudio.load(os.path.join(self.libri_root, feat_path))

            # 使用特征提取器提取特征
            feat = self.extracter(wav.squeeze())  # (seq_len, feat_dim)

            # print(f"feat.shape:{feat.shape}")

            return feat

    def __getitem__(self, index):
        # 加载声学特征和pad
        # X[index]: 第index个batch
        # x_file: 当前batch中一个数据的路径
        # x_batch: 存放当前batch所有音频数据进行特征提取后的特征，并对其进行随机采样，采样长度为sample_length
        # x_batch: (batch_size, seq_len, feat_dim)
        x_batch = [self._sample(self._load_feat(x_file)) for x_file in self.X[index]]

        # 将batch中所有特征长度一致(batch_size, seq_len, feat_dim)
        x_pad_batch = pad_sequence(x_batch, batch_first=True)

        return generate_masked_acoustic_model_data(spec=(x_pad_batch,), config=self.task_config,
                                                   score=self.X_score[index] if self.X_score is not None else None,
                                                   # awe=self.X_awe[index] if self.X_awe is not None else None,
                                                   # proportion=self.X_proportion[
                                                   #     index] if self.X_proportion is not None else None
                                                  )


# 使用Kaldi进行特征提取
class OnlineAcousticDataset(FeatDataset):

    def __init__(self, extracter, task_config, bucket_size, file_path, sets,
                 max_timestep=0, libri_root=None, target_level=-25, **kwargs):
        max_timestep *= 160
        super(OnlineAcousticDataset, self).__init__(extracter, task_config, bucket_size, file_path, sets,
                                                    max_timestep, libri_root, **kwargs)
        self.target_level = target_level
        self.sample_length = self.sample_length * 160

    def _normalize_wav_decibel(self, wav):
        """将音频标准化到目标分贝"""
        if self.target_level == 'None':
            return wav
        rms = wav.pow(2).mean().pow(0.5)
        scalar = (10 ** (self.target_level / 20)) / (rms + 1e-10)
        wav = wav * scalar
        return wav

    # 加载特征
    def _load_feat(self, feat_path):
        if self.libri_root is None:
            return torch.FloatTensor(np.load(os.path.join(self.root, feat_path)))
        else:
            wav, _ = torchaudio.load(os.path.join(self.libri_root, feat_path))
            wav = self._normalize_wav_decibel(wav.squeeze())
            return wav  # (seq_len)

    def _process_x_pad_batch(self, x_pad_batch):  # , x_score=None):
        if self.libri_root is not None:
            x_pad_batch = x_pad_batch.unsqueeze(1)  # (batch_size, channel=1, seq_len)
            feat_list = self.extracter(x_pad_batch)
            # print("########################")
            # print(feat_list.size)
            # print("########################")
        return generate_masked_acoustic_model_data(feat_list, config=self.task_config)  # , score=None)
        # x_score)

    def __getitem__(self, index):
        # 加载声学特征和pad
        # X[index]: 第index个batch
        # x_file: 当前batch中一个数据的路径
        # x_batch: 存放当前batch所有音频数据进行特征提取后的特征，并对其进行随机采样，采样长度为sample_length
        # x_batch: (batch_size, seq_len, feat_dim)
        x_batch = [self._sample(self._load_feat(x_file)) for x_file in self.X[index]]
        x_pad_batch = pad_sequence(x_batch, batch_first=True)
        # x_score = self.X_score[index]
        return self._process_x_pad_batch(x_pad_batch)  # , x_score=None)
        # x_score)
