import os
import random
import pandas as pd
from torch.utils.data.dataset import Dataset

HALF_BATCHSIZE_TIME = 99999


# FEAT DATASET
class FeatDataset(Dataset):

    # file_path: 当使用即时特征提取时，提供bucket长度 csv文件(, file_path, length, label)
    # libri_root: 数据集路径
    # sets: 存放三种数据集信息
    # bucket_size: 每个batch中填充的数据个数
    def __init__(self, extracter, task_config, bucket_size, file_path, sets,
                 max_timestep=0, libri_root=None, **kwargs):
        super(FeatDataset, self).__init__()

        # 提取器
        self.extracter = extracter
        self.task_config = task_config

        # LibirSpeech路径
        self.libri_root = libri_root

        # 采样的序列长度
        self.sample_length = task_config['sequence_length']

        # 采样长度大于0
        if self.sample_length > 0:
            print('[Dataset] - Sampling random segments for training, sample length:', self.sample_length)

        # 读取文件: csv存储(, file_path, length, label)
        self.root = file_path
        # 读取数据集信息
        tables = [pd.read_csv(os.path.join(file_path, s + '.csv')) for s in sets]

        # 将三种不同数据集中的数据信息整合，并按照长度进行降序排列
        self.table = pd.concat(tables, ignore_index=True).sort_values(by=['length'], ascending=False)
        print('[Dataset] - Training data from these sets:', str(sets))

        # 裁剪掉长度大于max_timestep的数据
        if max_timestep > 0:
            self.table = self.table[self.table.length < max_timestep]
        # 裁剪掉长度小于max_timestep的数据
        if max_timestep < 0:
            self.table = self.table[self.table.length > (-1 * max_timestep)]

        # 每个数据的路径信息
        X = self.table['file_path'].tolist()
        # 每个数据的序列长度信息
        X_lens = self.table['length'].tolist()
        X_scores = []  # 每个数据逐帧的情感分数:X_scores:(nums_wav, seq_len)
        X_awes = []  # 每个数据的声学词向量: (nums_wav, 240)
        X_proportions = []  # 每个数据的声学词向量比例: (nums_wav, word:proportion...)

        # 使用LibriSpeech数据集
        if 'train-clean-360' in sets:
            # 使用bucket允许在运行时存在不同的batch size
            # self.X: 存放所有batch, batch中存放的是数据的路径
            self.X = []
            self.X_score = []
            self.X_awe = None
            self.X_proportion = None

            X_score = self.table['score'].tolist()  # X_score: (nums_wav)
            for x_score in X_score:
                x_score = x_score.split(' ')
                del x_score[0]
                X_scores.append(x_score)

            # str->int
            X_scores = [[float(item) for item in x_score] for x_score in X_scores]
            batch_score = []

            # batch_x: 一个batch中所有数据的数据路径
            # batch_len: 一个batch中所有数据对应的长度
            batch_x, batch_len = [], []

            """加入情感分数"""
            for x, x_len, x_score in zip(X, X_lens, X_scores):
                batch_x.append(x)
                batch_len.append(x_len)
                batch_score.append(x_score)

                # batch_x中的数据达到了bucket_size的大小
                if len(batch_x) == bucket_size:
                    # 如果当前batch中数据的最大长度大于给定的HALF_BATCHSIZE_TIME，则将当前batch中的数据减半
                    # 即: 将一个batch拆分为两个batch
                    if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME) and self.sample_length == 0:
                        self.X.append(batch_x[:bucket_size // 2])
                        self.X.append(batch_x[bucket_size // 2:])
                        self.X_score.append(batch_score[:bucket_size // 2])
                        self.X_score.append(batch_score[bucket_size // 2:])
                    else:  # 长度不超出的话，当作一个batch
                        self.X.append(batch_x)
                        self.X_score.append(batch_score)
                    batch_x, batch_len, batch_score = [], [], []

            """未加入情感分数"""
            # for x, x_len in zip(X, X_lens):
            #     batch_x.append(x)
            #     batch_len.append(x_len)
            #
            #     # batch_x中的数据达到了bucket_size的大小
            #     if len(batch_x) == bucket_size:
            #         # 如果当前batch中数据的最大长度大于给定的HALF_BATCHSIZE_TIME，则将当前batch中的数据减半
            #         # 即: 将一个batch拆分为两个batch
            #         if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME) and self.sample_length == 0:
            #             self.X.append(batch_x[:bucket_size // 2])
            #             self.X.append(batch_x[bucket_size // 2:])
            #         else:  # 长度不超出的话，当作一个batch
            #             self.X.append(batch_x)
            #         batch_x, batch_len = [], []

            # 最后一个batch中的数据小于bucket_size，将其当作一个batch
            if len(batch_x) > 1:
                self.X.append(batch_x)
                self.X_score.append(batch_score)

        elif 'english' in sets:
            X_score = self.table['score'].tolist()  # X_score: (nums_wav)
            # X_awe = self.table['awe'].tolist()  # X_awe: (nums_wav)
            # X_proportion = self.table['proportion'].tolist()  # X_proportion: (nums_wav)

            for x_score in X_score:
                x_score = x_score.split(' ')
                del x_score[0]
                X_scores.append(x_score)

            # for x_awe in X_awe:
            #     X_awes.append(x_awe)
            #
            # for x_proportion in X_proportion:
            #     X_proportions.append(x_proportion)

            # str->int
            X_scores = [[float(item) for item in x_score] for x_score in X_scores]

            # 总计的样本数量
            self.num_samples = len(X)
            print('[Dataset] - Number of individual training instances:', self.num_samples)

            # 使用bucket允许在运行时存在不同的batch size
            # self.X: 存放所有batch, batch中存放的是数据的路径
            self.X = []

            # 存放所有batch, batch中存放的是数据的情感分数
            self.X_score = []

            # 存放所有的batch, batch存放的是数据的声学词向量
            self.X_awe = []

            # 存放所有的batch, batch存放的是数据的声学词向量比例
            self.X_proportion = []

            # batch_x: 一个batch中所有数据的数据路径
            # batch_len: 一个batch中所有数据对应的长度
            batch_x, batch_len = [], []

            # batch_score: 一个batch中所有数据的情感分数
            # batch_score: (batch_size, seq_length)
            batch_score = []
            # batch_awe = []
            # batch_proportion = []

            # X_scores:(nums_wav, seq_len)
            # for x, x_len, x_score, x_awe, x_proportion in zip(X, X_lens, X_scores, X_awes, X_proportions):
            for x, x_len, x_score in zip(X, X_lens, X_scores):
                batch_x.append(x)
                batch_len.append(x_len)
                batch_score.append(x_score)
                # batch_awe.append(x_awe)
                # batch_proportion.append(x_proportion)

                # batch_x中的数据达到了bucket_size的大小
                if len(batch_x) == bucket_size:
                    # 如果当前batch中数据的最大长度大于给定的HALF_BATCHSIZE_TIME，则将当前batch中的数据减半
                    # 即: 将一个batch拆分为两个batch
                    if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME) and self.sample_length == 0:
                        self.X.append(batch_x[:bucket_size // 2])
                        self.X.append(batch_x[bucket_size // 2:])
                        self.X_score.append(batch_score[:bucket_size // 2])
                        self.X_score.append(batch_score[bucket_size // 2:])
                        # self.X_awe.append(batch_awe[:bucket_size // 2])
                        # self.X_awe.append(batch_awe[bucket_size // 2:])
                        # self.X_proportion.append(batch_proportion[:bucket_size // 2])
                        # self.X_proportion.append(batch_proportion[bucket_size // 2:])
                    else:  # 长度不超出的话，当作一个batch
                        self.X.append(batch_x)
                        self.X_score.append(batch_score)
                        # self.X_awe.append(batch_awe)
                        # self.X_proportion.append(batch_proportion)
                    # batch_x, batch_len, batch_score, batch_awe, batch_proportion = [], [], [], [], []
                    batch_x, batch_len, batch_score = [], [], []

            # 最后一个batch中的数据小于bucket_size，将其当作一个batch
            if len(batch_x) > 1:
                self.X.append(batch_x)
                self.X_score.append(batch_score)
                # self.X_awe.append(batch_awe)
                # self.X_proportion.append(batch_proportion)

    # 对给定的数据x进行随机采样
    def _sample(self, x, score=None):
        if score is None:
            if self.sample_length <= 0:
                return x
            if len(x) < self.sample_length:
                return x

            # 对给定数据x进行随机采样，长度为sample_length
            idx = random.randint(0, len(x) - self.sample_length)
            return x[idx:idx + self.sample_length]
        else:
            if self.sample_length <= 0:
                return x, score
            if len(x) <= self.sample_length:
                # 判断特征与情感分数长度是否一致
                if x.size(0) > len(score):
                    diff = x.size(0) - len(score)
                    score += [0] * diff
                elif x.size(0) < len(score):
                    score = score[:x.size(0)]
                return x, score
            else:
                # 对给定数据x进行随机采样，长度为sample_length
                idx = random.randint(0, len(x) - self.sample_length - 1)
                return x[idx:idx + self.sample_length], score[idx:idx + self.sample_length]

    # 返回batch数量
    def __len__(self):
        return len(self.X)

    def collate_fn(self, items):
        items = items[0]  # hack bucketing
        return items


# WAVE DATASET
class WaveDataset(Dataset):

    def __init__(self, task_config, bucket_size, file_path, sets,
                 max_timestep=0, libri_root=None, **kwargs):
        super().__init__()

        self.task_config = task_config
        self.libri_root = libri_root

        # 采样的序列长度
        self.sample_length = task_config["sequence_length"]
        if self.sample_length > 0:
            print("[Dataset] - Sampling random segments for training, sample length:", self.sample_length, )

        # 读取文件: csv存储(, file_path, length, score, label)
        self.root = file_path
        # 读取数据集信息
        tables = [pd.read_csv(os.path.join(file_path, s + ".csv")) for s in sets]

        # 将三种不同数据集中的数据信息整合，并按照长度进行降序排列
        self.table = pd.concat(tables, ignore_index=True).sort_values(by=["length"], ascending=False)
        print("[Dataset] - Training data from these sets:", str(sets))

        # 裁剪掉长度大于max_timestep的数据
        if max_timestep > 0:
            self.table = self.table[self.table.length < max_timestep]
        # 裁剪掉长度小于max_timestep的数据
        if max_timestep < 0:
            self.table = self.table[self.table.length > (-1 * max_timestep)]

        # 每个数据的路径信息
        X = self.table["file_path"].tolist()
        # 每个数据的序列长度信息
        X_lens = self.table["length"].tolist()

        # 每个数据的情感分数
        X_scores = self.table['score'].tolist()

        # 样本数量
        self.num_samples = len(X)
        print("[Dataset] - Number of individual training instances:", self.num_samples)

        # 使用bucket允许在运行时存在不同的batch size
        # self.X: 存放所有batch，batch中存放的是数据的路径
        self.X = []

        # batch_x: 一个batch中所有数据的数据路径
        # batch_len: 一个batch中所有数据对应的长度
        batch_x, batch_len = [], []

        for x, x_len in zip(X, X_lens):
            batch_x.append(x)
            batch_len.append(x_len)

            # batch_x中的数据达到了bucket_size的大小
            if len(batch_x) == bucket_size:
                # 如果当前batch中数据的最大长度大于给定的HALF_BATCHSIZE_TIME，则将当前batch中的数据减半
                # 即: 将一个batch拆分为两个batch
                if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME) and self.sample_length == 0:
                    self.X.append(batch_x[: bucket_size // 2])
                    self.X.append(batch_x[bucket_size // 2:])
                else:  # 长度不超出的话，当作一个batch
                    self.X.append(batch_x)
                batch_x, batch_len = [], []

        # 最后一个batch中的数据小于bucket_size，将其当作一个batch
        if len(batch_x) > 1:
            self.X.append(batch_x)

    # 对给定数据x进行随机采样
    def _sample(self, x):
        if self.sample_length <= 0:
            return x
        if len(x) < self.sample_length:
            return x

        # 对给定数据x进行随机采样，长度为sample_length
        idx = random.randint(0, len(x) - self.sample_length)
        return x[idx: idx + self.sample_length]

    # 返回batch数量
    def __len__(self):
        return len(self.X)

    def collate_fn(self, items):
        items = items[0]  # hack bucketing
        assert (len(items) == 4), "__getitem__ should return (wave_input, wave_orig, wave_len, pad_mask)"
        return items
