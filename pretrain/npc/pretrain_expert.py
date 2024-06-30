# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ pretrain/npc/pretrain_expert.py ]
#   Synopsis     [ the NPC pretrain expert ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""

###############
# IMPORTATION #
###############
import copy
import torch
# -------------#
from utility.audio import plot_spectrogram_to_numpy
from pretrain.apc.pretrain_expert import UpstreamPretrainExpert as ApcPretrainExpert


####################
# UPSTREAM WRAPPER #
####################
class UpstreamPretrainExpert(ApcPretrainExpert):
    """
    The NPC pretrain expert
    """

    def __init__(self, datarc, upstream_config, device='cuda', multi_gpu=False, **kwargs):
        super(UpstreamPretrainExpert, self).__init__(datarc, upstream_config, device, multi_gpu, **kwargs)

    def _init_model(self):
        from upstream.npc.audio import create_transform
        from upstream.npc.npc import NPC

        try:
            print('[UpstreamPretrainExpert] - Using the apc preprocessor, on-the-fly feature preprocessing')
            preprocessor, feat_dim = create_transform(copy.deepcopy(self.upstream_config['data']['audio']))
        except:
            raise NotImplementedError(
                'Our upstream wrapper currently does not support other feature extracters, see: `s3prl/upstream/apc/expert.py`')

        print('[UpstreamPretrainExpert] - Initializing model...')
        self.model = NPC(feat_dim, **self.upstream_config["model"]["paras"])
        self.loss = torch.nn.L1Loss(reduction='none')
        return preprocessor

    # Interface
    def forward(self, data, records={}, global_step=0, log_step=1000, **kwargs):
        """
        Args:
            data:
                [spec_masked, pos_enc, mask_label, attn_mask, spec_target]
            
            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss        
        """

        audio_feat, audio_len = data[0], data[1]
        # 获取情感分数 BxSCORE
        score = data[2][0]
        # 获取声学词向量
        audio_awe = data[2][1]

        audio_awe = audio_awe.to(self.device)
        score = score.to(self.device)
        audio_feat = audio_feat.to(self.device)

        # 将情感分数扩充到BxSCORExC
        audio_score = torch.unsqueeze(score, 2)  # BxSCOREx1
        if audio_feat.size(1) == audio_score.size(1):
            audio_score = audio_score.expand_as(audio_feat)  # BxSCORExC
        else:
            padding_amount = audio_feat.shape[1] - audio_score.shape[1]
            audio_score = torch.nn.functional.pad(audio_score, (0, 0, 0, padding_amount), value=0)
            audio_score = audio_score.expand_as(audio_feat)
        audio_score = audio_score.to(self.device)

        # awe + score
        # audio_score = audio_score + audio_feat + audio_awe

        # score
        audio_score = audio_score + audio_feat

        # NPC: input = target
        # 原始模型
        # pred_spec, _ = self.model(audio_feat)

        # 包含情感分数与声学词向量
        # pred_spec, _ = self.model(audio_feat, score=score, awe=audio_awe, audio_score=audio_score)

        # 只包含情感分数，不包含声学词向量
        pred_spec, _ = self.model(audio_feat, score=score, awe=None, audio_score=audio_score)

        # 只包含声学词向量，不好含情感分数
        # pred_spec, _ = self.model(audio_feat, score=None, awe=audio_awe, audio_score=None)

        pred_feat = pred_spec[0]
        # pred_awe = pred_spec[1]
        pred_score = pred_spec[1]
        feat_loss = self.loss(pred_feat, audio_feat)
        # awe_loss = self.loss(pred_awe, audio_awe)
        score_loss = self.loss(pred_score, audio_score)

        # Compute loss on valid part only
        effective_feat_loss = 0
        # effective_awe_loss = 0
        effective_score_loss = 0
        for i, a_len in enumerate(audio_len):
            effective_feat_loss += feat_loss[i, :a_len, :].mean(dim=-1).sum()
            # effective_awe_loss += awe_loss[i, :a_len, :].mean(dim=-1).sum()
            effective_score_loss += score_loss[i, :a_len, :].mean(dim=-1).sum()
        feat_loss = effective_feat_loss / sum(audio_len)
        # awe_loss = effective_awe_loss / sum(audio_len)
        score_loss = effective_score_loss / sum(audio_len)

        if global_step % log_step == 0:
            spec_list = [pred_feat, audio_feat]
            name_list = ['pred_feat', 'true_spec']

            for i in range(len(spec_list)):
                spec = plot_spectrogram_to_numpy(spec_list[i][0].data.cpu().numpy())
                records[name_list[i]] = spec

        # loss = [feat_loss, awe_loss, score_loss]
        loss = [feat_loss, score_loss]
        # loss = [feat_loss, awe_loss]
        return loss, records
