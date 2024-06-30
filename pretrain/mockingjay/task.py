import copy
import torch
import random
import numpy as np
from functools import lru_cache

MAX_SEQLEN = 4500  # 最大sequence长度


@lru_cache(maxsize=128)
def get_sinusoid_table(hidden_size):  # Transformer中的正弦位置编码
    # 获取向量角度
    # position:位置 range(MAX_SEQLEN) hid_inx:hidden对应索引
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / hidden_size)

    # 获取hidden_size每个索引的位置向量角度
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(hidden_size)]  # dim: hidden_size

    # sinusoid_table:(MAX_SEQLEN, hidden_size)
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(MAX_SEQLEN)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)  # (MAX_SEQLEN, hidden_size)


# seq_len:序列长度 padding_idx:使得对应位置的  向量置为0
# 对table进行扩展
def fast_position_encoding(seq_len, hidden_size, batch_size=None, padding_idx=None):
    # 若当前序列长度大于给定的最大序列长度则返回异常
    assert seq_len <= MAX_SEQLEN, f'constant MAX_SEQLEN ({MAX_SEQLEN}) in mam.py < received seq_len ({seq_len})'

    # 获取上面的到的位置编码(seq_len, hidden_size)
    table = get_sinusoid_table(hidden_size)[:seq_len]

    if batch_size is not None:
        # 使用expand扩充
        batch_table = table.expand(batch_size, -1, -1)
        return batch_table  # (batch_size, seq_len, hidden_size)
    else:
        # 直接返回原来的table
        return table  # (seq_len, hidden_size)


def generate_masked_acoustic_model_data(spec, config, score=None, awe=None, proportion=None):
    """生成MAM数据"""

    with torch.no_grad():
        # 读取特征
        if len(spec) == 2:  # if self.duo_feature: dataloader will output `source_spec` and `target_spec`
            spec_masked = spec[0]
            spec_target = spec[1]
        elif len(spec) == 1:  # 默认
            # spec_masked == spec_target
            spec_masked = spec[0]  # (batch_size, seq_len, feat_dim)
            spec_target = copy.deepcopy(spec[0])  # (batch_size, seq_len, feat_dim)
        else:
            raise ValueError

        # 读取对应的情感分数(batch_size, seq_length)
        if score is None:
            spec_score = torch.zeros(spec_target.shape[0], spec_target.shape[1])
        else:
            spec_score = []

            # 获取spec_target长度
            length = spec_target[0].shape[0]

            # 获取score里面长度最长的列表
            max_score_length = max(len(item) for item in score)

            # 处理帧不一致情况
            for item in score:
                # 获取迭代list的长度
                item_length = len(item)

                item = [float(x) for x in item]
                # spce_target长度更长
                if length > max_score_length:
                    item += [0] * (length - item_length)
                else:
                    if length > item_length:
                        item += [0] * (length - item_length)
                    elif length < item_length:
                        item = item[0:length]
                item = torch.tensor(item)
                spec_score.append(item)

            spec_score = torch.stack(spec_score)
            spec_score = spec_score.expand(spec_masked.shape[0], spec_masked.shape[1])
            # for i in range(len(score)):
            #     print(f'spec_target: {spec_target[i].shape[0]}')
            #     print(f'score: {len(spec_score[i])}')

        # 读取对应的声学词向量
        if awe is None and proportion is None:
            spec_awe_target = torch.zeros(spec_target.shape[0], spec_target.shape[1], spec_target.shape[2])
        else:
            spec_awe_target = copy.deepcopy(spec_target)  # 对原始音频帧进行修改

            # 当前i表示batch中的第i个数据
            for i in range(len(awe)):
                cur_awe = awe[i].split(',')  # 当前音频对应的awe
                cur_proportion = proportion[i].split(',')  # 当前音频对应的proportion
                # 删除cur_awe最后一个元素, 对应的元素为空
                cur_awe.pop()
                cur_spec_awe_target = spec_awe_target[i]
                # 当前数据帧长度
                cur_frame_length = cur_spec_awe_target.shape[0]
                all_proportion = 100  # 比例总数
                past_frame_begin = 0  # 上一个单词开始的位置

                # 获取当前数据的所有单词、对应比例、240维的声学词向量
                for j in range(len(cur_awe)):
                    if j == len(cur_awe) - 1:
                        # 将剩余部分全部补充为0
                        cur_spec_awe_target[past_frame_begin:] = torch.zeros(240)
                    else:
                        awe_temp = cur_awe[j].split(':')  # 得到两部分 cur[0]:word, cur[1]:240维的声学词向量
                        proportion_temp = cur_proportion[j].split(':')  # 同样得到两部分 cur[0]:word, cur[1]:当前单词对应的比例
                        float_awe = awe_temp[1].split(' ')  # 240维声学词向量
                        float_awe = [float(i) for i in float_awe]

                        # 计算比例
                        proportion_temp[1] = int(proportion_temp[1])
                        # 转为int
                        cur_word_proportion = int(cur_frame_length * proportion_temp[1] * 0.01)
                        all_proportion -= cur_word_proportion
                        cur_spec_awe_target[past_frame_begin:past_frame_begin + cur_word_proportion] = torch.tensor(
                            float_awe)
                        past_frame_begin += cur_word_proportion

        spec_awe_masked = copy.deepcopy(spec_awe_target)
        spec_score_target = copy.deepcopy(spec_score)

        # spec_target.sum(dim=-1): (batch_size, seq_len)
        # spec_target.sum(dim=-1) != 0).long(): (batch_size, seq_len), 除了pad的位置其余全为1
        # spec_len: 存储每一个utterance的长度 (batch_size)
        spec_len = (spec_target.sum(dim=-1) != 0).long().sum(dim=-1).tolist()
        # print(spec_len)
        batch_size = spec_target.shape[0]
        seq_len = spec_target.shape[1]

        # 位置编码
        # pos_enc: (seq_len, position_encoding_size)
        pos_enc = fast_position_encoding(seq_len, config['position_encoding_size'])

        # mask_proportion: mask概率 mockingjay为15%
        # mask_frequency:
        if config['mask_proportion'] != 0 or config['mask_frequency'] != 0:
            # Default
            # mask_label: 生成和spec_target相同维度但内容全为0的tensor (batch_size, seq_len, feat_dim)
            mask_label = torch.zeros_like(spec_target, dtype=torch.uint8)
        else:
            # mask_label: 生成和spec_target相同维度但内容全为1的tensor (batch_size, seq_len, feat_dim)
            mask_label = torch.ones_like(spec_target, dtype=torch.uint8)

        # attn_mask: (batch_size, seq_len)
        attn_mask = torch.ones((batch_size, seq_len))

        for idx in range(batch_size):
            # 将当前数据pad的部分置为0
            attn_mask[idx, spec_len[idx]:] = 0

            # 为了避免模型利用声学帧的局部平滑性，将连续帧屏蔽为0
            # starts: 屏蔽连续帧所有的起始位置,共包含proportion个
            # consecutive: 屏蔽连续帧的数量, mockingjay默认为7
            def _starts_to_intervals(starts, consecutive):

                # tiled: (proportion, consecutive)
                tiled = starts.expand(consecutive, starts.size(0)).permute(1, 0)
                offset = torch.arange(consecutive).expand_as(tiled)
                intervals = tiled + offset
                return intervals.view(-1)

            config['mask_proportion'] = 0.15
            # time masking
            if config['mask_proportion'] > 0:
                # mask_consecutive: mask连续帧的数量
                mask_consecutive = random.randint(config['mask_consecutive_min'], config['mask_consecutive_max'])

                # valid_start_max: 连续帧初始帧的最大位置
                valid_start_max = max(spec_len[idx] - mask_consecutive - 1, 0)

                # mask连续帧的比例
                proportion = round(spec_len[idx] * config['mask_proportion'] / mask_consecutive)

                # mask_allow_overlap: 是否允许重叠mask,默认true
                if config['mask_allow_overlap']:  # 允许重叠mask
                    # 从范围 (0, valid_index_range) 中抽取“比例”样本且不进行替换
                    # chose_starts: 连续屏蔽帧的起始位置，长度为proportion

                    # 屏蔽的最小情感分数
                    # mask_score = config['mask_score']
                    # mask_score = 0.65

                    chosen_starts = []
                    # 首先进行随机打乱
                    begin_index = torch.randperm(valid_start_max + 1).tolist()

                    # 随机打乱应该换成按照帧情感分数排序
                    _, max_index = torch.topk(spec_score[idx], k=proportion * 2)
                    # 选取其中符合初始索引的索引
                    for index in max_index:
                        if index in begin_index and len(chosen_starts) < proportion:
                            chosen_starts.append(index)

                    # for index in begin_index:
                    #     if spec_score[idx, index] > mask_score:
                    #         chosen_starts.append(index)
                    #         if len(chosen_starts) == proportion:
                    #             break

                    # 当前语音的情感分数过低
                    if len(chosen_starts) == proportion:
                        chosen_starts = torch.tensor(chosen_starts)
                    else:
                        chosen_starts = torch.randperm(valid_start_max + 1)[:proportion]
                else:
                    mask_bucket_size = round(mask_consecutive * config['mask_bucket_ratio'])
                    rand_start = random.randint(0, min(mask_consecutive, valid_start_max))
                    valid_starts = torch.arange(rand_start, valid_start_max + 1, mask_bucket_size)
                    chosen_starts = valid_starts[torch.randperm(len(valid_starts))[:proportion]]

                # chosen_intervals: 存放要屏蔽帧的所有位置 (proportion * consecutive)
                chosen_intervals = _starts_to_intervals(chosen_starts, mask_consecutive).type(torch.long)

                # 决定是mask / random / do nothing
                dice = random.random()

                # 80%概率直接屏蔽为0
                if dice < 0.8:
                    spec_masked[idx, chosen_intervals, :] = 0
                    spec_score[idx, chosen_intervals] = 0
                    spec_awe_masked[idx, chosen_intervals, :] = 0
                # 10%概率替换为其他随机帧
                elif 0.8 <= dice < 0.9:
                    random_starts = torch.randperm(valid_start_max + 1)[:proportion]
                    random_intervals = _starts_to_intervals(random_starts, mask_consecutive)
                    spec_masked[idx, chosen_intervals, :] = spec_masked[idx, random_intervals, :]
                    spec_score[idx, chosen_intervals] = spec_score[idx, random_intervals]
                    spec_awe_masked[idx, chosen_intervals, :] = spec_awe_masked[idx, random_intervals, :]
                # 10%概率do nothing
                else:
                    pass

                # the gradients will be calculated on chosen frames
                # 所有被mask的帧置为1,计算梯度
                mask_label[idx, chosen_intervals, :] = 1

            # frequency masking
            # mask_frequency: mask maximum this percentage of frequency bands, set to 0 for no frequency mask
            if config['mask_frequency'] > 0:
                # spce_masked: (batch_size, seq_length, feat_dim)
                # mask最长宽度
                max_width = int(spec_masked.shape[2] * config['mask_frequency'])

                rand_bandwidth = random.randint(0, max_width)
                chosen_starts = torch.randperm(spec_masked.shape[2] - rand_bandwidth)[:1]
                chosen_intervals = _starts_to_intervals(chosen_starts, rand_bandwidth)
                spec_masked[idx, :, chosen_intervals] = 0

                # the gradients will be calculated on chosen frames
                mask_label[idx, :spec_len[idx], chosen_intervals] = 1

        # noise_proportion > 0: 在训练时加入高斯噪音
        if config['noise_proportion'] > 0:
            # noise augmentation 高斯噪音
            dice = random.random()
            if dice < config['noise_proportion']:
                noise_sampler = torch.distributions.Normal(0, 0.2)
                spec_masked += noise_sampler.sample(spec_masked.shape).to(device=spec_masked.device)

        # 将情感分数与mask相加
        # (batch_size, seq_len, 1)
        spec_score = torch.unsqueeze(spec_score, 2)
        spec_score = spec_score.expand_as(spec_masked)

        spec_masked = spec_masked + spec_score + spec_awe_masked

        # 原始数据
        valid_batchid = mask_label.view(batch_size, -1).sum(dim=-1).nonzero(as_tuple=False).view(
            -1)  # valid_batchid: 被mask的batch_id ()
        spec_masked = spec_masked.to(dtype=torch.float32)[valid_batchid]
        spec_target = spec_target.to(dtype=torch.float32)[valid_batchid]

        # 情感分数
        spec_score = spec_score.to(dtype=torch.float32)[valid_batchid]
        spec_score_target = torch.unsqueeze(spec_score_target, 2)
        spec_score_target = spec_score_target.expand_as(spec_target)
        spec_score_target = spec_score_target.to(dtype=torch.float32)[valid_batchid]

        # 声学词向量
        spec_awe_masked = spec_awe_masked.to(dtype=torch.float32)[valid_batchid]
        spec_awe_target = spec_awe_target.to(dtype=torch.float32)[valid_batchid]

        # 返回数据
        attn_mask = attn_mask.to(dtype=torch.float32)[valid_batchid]
        mask_label = mask_label.to(dtype=torch.bool)[
            valid_batchid]  # mask_label: (batch_size, seq_len, feat_dim),被mask的帧为1,其余为0
        pos_enc = pos_enc.to(dtype=torch.float32)
        spec_masked_all = [spec_masked, spec_score, spec_awe_masked]
        spec_target_all = [spec_target, spec_score_target, spec_awe_target]

    return spec_masked_all, pos_enc, mask_label, attn_mask, spec_target_all
