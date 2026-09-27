"""FireRedPunc in pure MLX: BERT token classifier + punctuation rules.

Mirrors fireredpunc/punc.py (default config: sentence_max_length=-1 → plain
argmax path). text rules and timestamp sentence assembly are verbatim ports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .bert_mlx import bert_forward
from .tokenizer_punc import BertWordPieceTokenizer
from .token_dict_punc import TokenDict


@dataclass
class FireRedPuncMlxConfig:
    sentence_max_length: int = -1
    # CLI --punc_with_timestamp: 0 forces the plain-text punc path even when
    # ASR word timestamps are available (official flag, no-op in the official
    # 4-module system; wired up here so the MLX CLI honours it).
    punc_with_timestamp: int = 1


class RuleBaedTxtFix:
    @classmethod
    def fix(cls, txt_ori, capitalize_first=True):
        txt = txt_ori.lower()
        txt = re.sub(r"([a-z])，([a-z])", r"\1, \2", txt)
        txt = re.sub(r"([a-z])。([a-z])", r"\1. \2", txt)
        txt = re.sub(r"([a-z])？([a-z])", r"\1? \2", txt)
        txt = re.sub(r"([a-z])！([a-z])", r"\1! \2", txt)
        txt = re.sub(r"^([a-z]+)，", r"\1,", txt)
        txt = re.sub(r"^([a-z]+)。", r"\1.", txt)
        txt = re.sub(r"^([a-z]+)？", r"\1?", txt)
        txt = re.sub(r"^([a-z]+)！", r"\1!", txt)
        txt = re.sub(r"( [a-zA-Z']+)，$", r"\1,", txt)
        txt = re.sub(r"( [a-zA-Z']+)。$", r"\1.", txt)
        txt = re.sub(r"( [a-zA-Z']+)？$", r"\1?", txt)
        txt = re.sub(r"( [a-zA-Z']+)！$", r"\1!", txt)
        txt = re.sub("^i ", "I ", txt)
        txt = re.sub("^i'm ", "I'm ", txt)
        txt = re.sub("^i'd ", "I'd ", txt)
        txt = re.sub("^i've ", "I've ", txt)
        txt = re.sub("^i'll ", "I'll ", txt)
        txt = re.sub(" i ", " I ", txt)
        txt = re.sub(" i'm ", " I'm ", txt)
        txt = re.sub(" i'd ", " I'd ", txt)
        txt = re.sub(" i've ", " I've ", txt)
        txt = re.sub(" i'll ", " I'll ", txt)
        if capitalize_first and len(txt) > 0 and re.match("[a-z]", txt[0]):
            txt = txt[0].upper() + txt[1:]
        txt = re.sub(r'([.!?。？！])\s+([a-z])',
                     lambda m: f"{m.group(1)} {m.group(2).upper()}", txt)
        return txt


class FireRedPuncMlx:
    @classmethod
    def from_pretrained(cls, model_dir, config=FireRedPuncMlxConfig()):
        from ..loader import load_safetensors_plain
        model_dir = Path(model_dir)
        w, args = load_safetensors_plain(model_dir)
        tokenizer = BertWordPieceTokenizer(
            str(model_dir / "chinese-lert-base" / "vocab.txt"))
        in_dict = TokenDict(model_dir / "chinese-bert-wwm-ext_vocab.txt", unk="[UNK]")
        out_dict = TokenDict(model_dir / "out_dict")
        cls_id = args["cls_id"]
        return cls(w, args, tokenizer, in_dict, out_dict, cls_id, config)

    def __init__(self, w, args, tokenizer, in_dict, out_dict, cls_id, config):
        self.w, self.args = w, args
        self.tokenizer, self.in_dict, self.out_dict = tokenizer, in_dict, out_dict
        self.cls_id, self.config = cls_id, config
        self.INPUT_IGNORE_ID = self.in_dict["[PAD]"]
        self.DEFAULT_OUT = " "

    # ---- io ----
    def text2tensor(self, batch_text):
        batch_txt_tokens, batch_seqs = [], []
        for text in batch_text:
            tokens, _ = self.tokenizer.tokenize(text, recover_unk=True)
            batch_txt_tokens.append(tokens)
            batch_seqs.append([self.in_dict.get(t, self.in_dict.unk)
                               for t in tokens])
        return (*self._pad(batch_seqs), batch_txt_tokens)

    def timestamp2tensor(self, batch_timestamp):
        batch_txt_tokens, batch_seqs, batch_split = [], [], []
        for timestamps in batch_timestamp:
            txt_token, seq, split = [], [], []
            for token, start, end in timestamps:
                sub_tokens, _ = self.tokenizer.tokenize(token, recover_unk=True)
                split.append(len(sub_tokens))
                txt_token.extend(sub_tokens)
                seq.extend(self.in_dict.get(t, self.in_dict.unk)
                           for t in sub_tokens)
            batch_txt_tokens.append(txt_token)
            batch_seqs.append(seq)
            batch_split.append(split)
        return (*self._pad(batch_seqs), batch_txt_tokens, batch_split)

    def _pad(self, seqs):
        lengths = [len(s) for s in seqs]
        T = max(lengths)
        padded = np.full((len(seqs), T), self.INPUT_IGNORE_ID, dtype=np.int64)
        for i, s in enumerate(seqs):
            padded[i, :len(s)] = s
        return mx.array(padded), lengths

    # ---- inference ----
    def logits(self, padded, lengths):
        ids = np.asarray(padded).copy()
        # prepend [CLS] per official add_cls
        cls_col = np.full((ids.shape[0], 1), self.cls_id, dtype=np.int64)
        ids = np.concatenate([cls_col, ids], axis=1)
        lengths2 = [l + 1 for l in lengths]
        out = bert_forward(self.w, mx.array(ids), lengths2)
        scores = out[:, 1:, :]  # drop CLS position
        logits = scores @ self.w["classifier.weight"].T \
            + self.w["classifier.bias"]
        return logits, lengths

    def get_punc_pred(self, logits, lengths):
        preds = np.asarray(mx.argmax(logits, axis=-1))
        return [pred[:lengths[i]].tolist() for i, pred in enumerate(preds)]

    # ---- public API (mirrors official FireRedPunc) ----
    def process(self, batch_text, batch_uttid=None):
        if not batch_text:
            return []
        padded, lengths, txt_tokens = self.text2tensor(batch_text)
        logits, lengths = self.logits(padded, lengths)
        preds = self.get_punc_pred(logits, lengths)
        punc_txts = self.add_punc_to_txt(txt_tokens, preds)
        punc_txts = [RuleBaedTxtFix.fix(t) for t in punc_txts]
        results = []
        for i in range(len(batch_text)):
            r = {"punc_text": punc_txts[i], "origin_text": batch_text[i]}
            if batch_uttid is not None:
                r["uttid"] = batch_uttid[i]
            results.append(r)
        return results

    def process_with_timestamp(self, batch_timestamp, batch_uttid=None):
        if not batch_timestamp:
            return []
        padded, lengths, batch_txt_tokens, batch_split = self.timestamp2tensor(batch_timestamp)
        logits, lengths = self.logits(padded, lengths)
        preds = self.get_punc_pred(logits, lengths)
        punc_txts = self.add_punc_to_txt_with_timestamp(
            batch_txt_tokens, preds, batch_timestamp, batch_split)
        new_punc_txts = []
        for txts in punc_txts:
            new_txts = []
            for idx, txt in enumerate(txts):
                if idx == 0:
                    cap = True
                else:
                    prev_text = new_txts[idx - 1][0]
                    cap = bool(prev_text) and prev_text[-1] in '.!?。？！'
                new_txts.append((RuleBaedTxtFix.fix(txt[0], capitalize_first=cap),
                                 txt[1], txt[2]))
            new_punc_txts.append(new_txts)
        results = []
        for i in range(len(batch_timestamp)):
            r = {"punc_sentences": [
                {"punc_text": t[0], "start_s": t[1], "end_s": t[2]}
                for t in new_punc_txts[i]]}
            if batch_uttid is not None:
                r["uttid"] = batch_uttid[i]
            results.append(r)
        return results

    # ---- text rules (verbatim) ----
    def add_punc_to_txt(self, token_seqs, pred_seqs):
        punc_txts = []
        for token_seq, pred_seq in zip(token_seqs, pred_seqs):
            assert len(token_seq) == len(pred_seq)
            txt = ""
            for i, token in enumerate(token_seq):
                tag = self.out_dict[pred_seq[i]]
                if token.startswith("##"):
                    token = token.replace("##", "")
                elif re.search("[a-zA-Z0-9#]+", token) and \
                        i > 0 and re.search("[a-zA-Z0-9#]+", token_seq[i - 1]):
                    if self.out_dict[pred_seq[i - 1]] == self.DEFAULT_OUT:
                        token = " " + token
                txt += token if tag == self.DEFAULT_OUT else token + tag
            txt = txt.replace("  ", " ")
            punc_txts.append(txt)
        return punc_txts

    def add_punc_to_txt_with_timestamp(self, token_seqs, pred_seqs,
                                       batch_timestamp, batch_tokens_split_num):
        punc_txts = []
        for token_seq, pred_seq, timestamps, tokens_split_num in \
                zip(token_seqs, pred_seqs, batch_timestamp,
                    batch_tokens_split_num):
            assert len(token_seq) == len(pred_seq)
            sentences = []
            txt, start, end = "", -1, -1
            i = j = 0
            last_token = last_tag = ""
            while i < len(token_seq) and j < len(tokens_split_num):
                split_num = tokens_split_num[j]
                timestamp = timestamps[j]
                assert len(timestamp) == 3
                if start == -1:
                    start = timestamp[1]
                end = timestamp[2]
                token = ""
                tag = self.DEFAULT_OUT
                for k in range(split_num):
                    sub_token = token_seq[i]
                    tag = self.out_dict[pred_seq[i]]
                    sub_token = re.sub("^##", "", sub_token)
                    token = sub_token if k == 0 else token + sub_token
                    i += 1
                if split_num == 0:
                    token = timestamp[0]
                assert token == timestamp[0], f"{token}/{timestamp}"
                j += 1
                if re.search("[a-zA-Z0-9#]+", token) and \
                        j > 0 and re.search("[a-zA-Z0-9#]+", last_token):
                    if last_tag == self.DEFAULT_OUT:
                        token = " " + token
                if tag == self.DEFAULT_OUT:
                    txt += token
                else:
                    txt += token + tag
                    txt = txt.replace("  ", " ")
                    assert start != -1
                    sentences.append((txt, start, end))
                    txt, start, end = "", -1, -1
                last_token = token
                last_tag = tag
            if txt != "":
                assert start != -1 and end != -1
                sentences.append((txt, start, end))
            punc_txts.append(sentences)
        return punc_txts
