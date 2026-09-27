"""Torch-free re-implementation of the official ASRFeatExtractor.

Identical numerics to fireredasr2s/fireredasr2/data/asr_feat.py (kaldi
fbank + CMVN), with torch tensor conversion removed: everything stays
numpy until the facade hands mx.array in.
"""

from __future__ import annotations

import math
import os

import kaldi_native_fbank as knf
import kaldiio
import numpy as np


class ASRFeatExtractor:
    def __init__(self, kaldi_cmvn_file):
        self.cmvn = CMVN(kaldi_cmvn_file) if kaldi_cmvn_file != "" else None
        self.fbank = KaldifeatFbank(num_mel_bins=80, frame_length=25,
                                    frame_shift=10, dither=0.0)

    def __call__(self, wav_paths, wav_uttids):
        feats, durs, paths, uttids = [], [], [], []
        wav_datas = []
        if isinstance(wav_paths[0], str):
            for wav_path in wav_paths:
                wav_datas.append(kaldiio.load_mat(wav_path))
        else:
            wav_datas = wav_paths
        for (sample_rate, wav_np), path, uttid in zip(wav_datas, wav_paths,
                                                      wav_uttids):
            dur = wav_np.shape[0] / sample_rate
            fbank = self.fbank((sample_rate, wav_np))
            if fbank.shape[0] < 1:
                continue
            if self.cmvn is not None:
                fbank = self.cmvn(fbank)
            feats.append(fbank.astype(np.float32))
            durs.append(dur)
            paths.append(path)
            uttids.append(uttid)
        if feats:
            lengths = np.array([f.shape[0] for f in feats], dtype=np.int64)
            feats_pad = pad_feat(feats, 0.0)
        else:
            lengths, feats_pad = None, None
        return feats_pad, lengths, durs, paths, uttids


def pad_feat(xs, pad_value):
    n = len(xs)
    max_len = max(x.shape[0] for x in xs)
    out = np.full((n, max_len, xs[0].shape[1]), pad_value, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        out[i, :x.shape[0]] = x
    return out


class CMVN:
    def __init__(self, kaldi_cmvn_file):
        self.dim, self.means, self.inverse_std_variences = \
            self.read_kaldi_cmvn(kaldi_cmvn_file)

    def __call__(self, x, is_train=False):
        assert x.shape[-1] == self.dim, "CMVN dim mismatch"
        return (x - self.means) * self.inverse_std_variences

    def read_kaldi_cmvn(self, kaldi_cmvn_file):
        assert os.path.exists(kaldi_cmvn_file)
        stats = kaldiio.load_mat(kaldi_cmvn_file)
        assert stats.shape[0] == 2
        dim = stats.shape[-1] - 1
        count = stats[0, dim]
        assert count >= 1
        floor = 1e-20
        means = np.zeros(dim, dtype=np.float64)
        istd = np.zeros(dim, dtype=np.float64)
        for d in range(dim):
            mean = stats[0, d] / count
            var = (stats[1, d] / count) - mean * mean
            means[d] = mean
            istd[d] = 1.0 / math.sqrt(max(var, floor))
        return dim, means, istd


class KaldifeatFbank:
    def __init__(self, num_mel_bins=80, frame_length=25, frame_shift=10,
                 dither=1.0):
        self.dither = dither
        opts = knf.FbankOptions()
        opts.frame_opts.dither = dither
        opts.mel_opts.num_bins = num_mel_bins
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False
        self.opts = opts

    def __call__(self, wav, is_train=False):
        if isinstance(wav, str):
            sample_rate, wav_np = kaldiio.load_mat(wav)
        elif isinstance(wav, (tuple, list)) and len(wav) == 2:
            sample_rate, wav_np = wav
        assert len(wav_np.shape) == 1
        self.opts.frame_opts.dither = self.dither if is_train else 0.0
        fbank = knf.OnlineFbank(self.opts)
        fbank.accept_waveform(sample_rate, wav_np.tolist())
        feat = [fbank.get_frame(i) for i in range(fbank.num_frames_ready)]
        if not feat:
            return np.zeros((0, self.opts.mel_opts.num_bins))
        return np.vstack(feat)
