# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ pretrain/apc/dataset.py ]
#   Synopsis     [ the dataset that applies the apc preprocessing on audio ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
import copy

"""*********************************************************************************************"""

###############
# IMPORTATION #
###############
import os
import numpy as np
# -------------#
import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
# -------------#
from pretrain.bucket_dataset import FeatDataset


class ApcAudioDataset(FeatDataset):

    def __init__(self, extracter, task_config, bucket_size, file_path, sets,
                 max_timestep=0, libri_root=None, **kwargs):
        super(ApcAudioDataset, self).__init__(extracter, task_config, bucket_size, file_path, sets,
                                              max_timestep, libri_root, **kwargs)

    def _load_feat(self, feat_path):
        if self.libri_root is None:
            return torch.FloatTensor(np.load(os.path.join(self.root, feat_path)))
        else:
            wav, _ = torchaudio.load(os.path.join(self.libri_root, feat_path))
            feat = self.extracter(wav)
            return feat

    def __getitem__(self, index):
        # Load acoustic feature and pad
        emotion_info = list()
        if self.X_score is None:
            x_batch = [self._sample(self._load_feat(x_file)) for x_file in self.X[index]]
            x_len = [len(x_b) for x_b in x_batch]
            x_pad_batch = pad_sequence(x_batch, batch_first=True)
            emotion_info = [None, None]
        else:
            list_x_batch = []
            list_score = []
            # 获取特征、情感分数，并对其进行随机采样
            for x_file, x_score in zip(self.X[index], self.X_score[index]):
                x, score = self._sample(self._load_feat(x_file), x_score)
                list_x_batch.append(x)
                list_score.append(torch.tensor(score))
            # BxTxC
            x_pad_batch = pad_sequence(list_x_batch, batch_first=True)
            # BxSCORE
            x_pad_score = pad_sequence(list_score, batch_first=True)

            x_len = [len(x_b) for x_b in list_x_batch]
            emotion_info.append(x_pad_score)

            # 读取对应的声学词向量
            awe = self.X_awe[index]
            proportion = self.X_proportion[index]
            awe_batch = copy.deepcopy(x_pad_batch)  # 对原始的音频帧进行修改
            # 读取当前batch中的每个数据
            for i in range(len(awe)):
                # 当前音频对应的awe
                cur_awe = awe[i].split(',')
                # 当前音频对应的proportion
                cur_proportion = proportion[i].split(',')
                # 删除当前awe的最后一个元素，对应的元素为空
                cur_awe.pop()
                # 读取当前音频对应的特征
                cur_awe_target = awe_batch[i]
                # 当前音频帧的长度
                cur_frame_length = cur_awe_target.shape[0]
                # 比例总数
                all_proportion = 100
                # 上一个单词开始的位置
                past_word_begin = 0

                # 获取当前音频对应的所有单词、各个单词比例以及80维的声学词向量
                for j in range(len(cur_awe)):
                    # 如果当前是最后一个单词，将剩余部分全部补充为0
                    if j == len(cur_awe) - 1:
                        cur_awe_target[past_word_begin:, :] = torch.zeros(80)
                    else:
                        # 得到两部分: awe_temp[0]:word, awe_temp[1]:80维的声学词向量
                        awe_temp = cur_awe[j].split(':')
                        # 得到两部分: proportion_temp[0]:word, proportion_temp[1]:每个单词对应的比例
                        proportion_temp = cur_proportion[j].split(':')
                        # 得到所有的80维的声学词向量
                        float_awe = awe_temp[1].split(' ')
                        float_awe = [float(x) for x in float_awe]
                        # 计算每个单词的比例
                        proportion_temp[1] = int(proportion_temp[1])
                        cur_word_proportion = int(cur_frame_length * proportion_temp[1] * 0.01)
                        all_proportion -= cur_word_proportion
                        cur_awe_target[past_word_begin:past_word_begin + cur_word_proportion, :] = torch.tensor(
                            float_awe
                        )
                        past_word_begin += cur_word_proportion
            emotion_info.append(awe_batch)
        return x_pad_batch, x_len, emotion_info

