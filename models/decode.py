import os
import pdb
import time
import torch
import ctcdecode
import numpy as np
from itertools import groupby
import torch.nn.functional as F


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}

        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        
        self.ctc_decoder = ctcdecode.CTCBeamDecoder(
            [chr(x) for x in range(20000, 20000 + num_classes)], 
            beam_width=10,
            blank_id=blank_id,
            num_processes=10
        )

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        
        return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        '''
        CTCBeamDecoder Shape:
            - Input:  nn_output (B, T, N), which should be passed through a softmax layer
            - Output: beam_resuls (B, N_beams, T), int, need to be decoded by i2g_dict
                        scores (B, N_beams), p=1/np.exp(beam_score)
                        t (B, N_beams)
                        out_lens (B, N_beams)
        '''
        ret_list = []
        vid_lgt = vid_lgt.cpu()
        if not probs: nn_output = nn_output.softmax(-1).cpu()
        res, scores, t, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)

        for batch_idx in range(len(nn_output)):
            first_result = res[batch_idx][0][:out_seq_len[batch_idx][0]]

            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])

            ret_list.append([(
                self.i2g_dict[int(gloss_id)], idx) 
                for idx, gloss_id in enumerate(first_result)
            ])

        return ret_list