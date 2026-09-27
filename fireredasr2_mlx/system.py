"""FireRedASR2System in pure MLX: VAD → LID → ASR → Punc pipeline.

Verbatim port of fireredasr2s/fireredasr2system.py process() using the MLX
module implementations. All fields of the official result dict are preserved.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import soundfile as sf

from . import FireRedAsr2Mlx, FireRedAsr2MlxConfig
from .llm import FireRedAsr2LlmMlx, FireRedAsr2LlmMlxConfig
from .models.lid import FireRedLidMlx, FireRedLidMlxConfig
from .models.vad import FireRedVadMlx, FireRedVadMlxConfig
from .punc.punc_mlx import FireRedPuncMlx, FireRedPuncMlxConfig

logger = logging.getLogger("fireredasr2s.asr_system")


@dataclass
class FireRedAsr2SystemMlxConfig:
    vad_model_dir: str = "pretrained_models/FireRedVAD/VAD"
    lid_model_dir: str = "pretrained_models/FireRedLID"
    asr_model_dir: str = "pretrained_models/FireRedASR2-AED"
    punc_model_dir: str = "pretrained_models/FireRedPunc"
    asr_type: str = "aed"
    vad_config: FireRedVadMlxConfig = field(default_factory=FireRedVadMlxConfig)
    lid_config: FireRedLidMlxConfig = field(default_factory=FireRedLidMlxConfig)
    asr_config: FireRedAsr2MlxConfig = field(default_factory=FireRedAsr2MlxConfig)
    llm_config: FireRedAsr2LlmMlxConfig = field(default_factory=FireRedAsr2LlmMlxConfig)
    punc_config: FireRedPuncMlxConfig = field(default_factory=FireRedPuncMlxConfig)
    asr_batch_size: int = 1
    punc_batch_size: int = 1
    enable_vad: bool = True
    enable_lid: bool = True
    enable_punc: bool = True


class FireRedAsr2SystemMlx:
    def __init__(self, config: FireRedAsr2SystemMlxConfig):
        c = config
        if c.asr_type == "llm":
            # The LLM variant has no word timestamps; force the no-timestamp
            # assembly path (the official 4-module CLI crashes here instead).
            c.asr_config.return_timestamp = False
        self.asr_type = c.asr_type
        self.vad = FireRedVadMlx.from_pretrained(c.vad_model_dir, c.vad_config) \
            if c.enable_vad else None
        self.lid = FireRedLidMlx.from_pretrained(c.lid_model_dir, c.lid_config) \
            if c.enable_lid else None
        if c.asr_type == "llm":
            self.asr = FireRedAsr2LlmMlx.from_pretrained(c.asr_model_dir,
                                                         c.llm_config)
        else:
            self.asr = FireRedAsr2Mlx.from_pretrained(c.asr_model_dir,
                                                      c.asr_config)
        self.punc = FireRedPuncMlx.from_pretrained(c.punc_model_dir, c.punc_config) \
            if c.enable_punc else None
        self.config = config

    def process(self, wav_path, uttid="tmpid"):
        wav_np, sample_rate = sf.read(wav_path, dtype="int16")
        dur = wav_np.shape[0] / sample_rate

        # 1. VAD
        if self.config.enable_vad:
            vad_result, _ = self.vad.detect(wav_path)
            vad_segments = vad_result["timestamps"]
            logger.info(f"VAD: {vad_result}")
        else:
            vad_segments = [(0, dur)]
            vad_result = {"timestamps": vad_segments}

        # 2-3. per-segment ASR (+LID)
        asr_results, lid_results = [], []
        batch_asr_uttid, batch_asr_wav = [], []
        for j, (start_s, end_s) in enumerate(vad_segments):
            wav_segment = wav_np[int(start_s * sample_rate):int(end_s * sample_rate)]
            vad_uttid = f"{uttid}_s{int(start_s * 1000)}_e{int(end_s * 1000)}"
            batch_asr_uttid.append(vad_uttid)
            batch_asr_wav.append((sample_rate, wav_segment))
            if len(batch_asr_uttid) < self.config.asr_batch_size \
                    and j != len(vad_segments) - 1:
                continue

            batch_asr_results = self.asr.transcribe(batch_asr_uttid, batch_asr_wav)
            logger.info(f"ASR: {batch_asr_results}")

            if self.config.enable_lid:
                batch_lid_results = self.lid.process(batch_asr_uttid, batch_asr_wav)
                logger.info(f"LID: {batch_lid_results}")
            else:
                batch_lid_results = [None] * len(batch_asr_results)

            for a_res, l_res in zip(batch_asr_results, batch_lid_results):
                text = a_res.get("text", "").strip()
                if not text or re.search(r"(<blank>)|(<sil>)", text):
                    continue
                asr_results.append(a_res)
                lid_results.append(l_res)

            batch_asr_uttid, batch_asr_wav = [], []

        # 4-5. Punc
        # Timestamped punc needs both ASR word timestamps and
        # --punc_with_timestamp 1; otherwise fall back to plain-text punc.
        with_ts = bool(self.config.asr_config.return_timestamp) and \
            bool(self.config.punc_config.punc_with_timestamp)
        if self.config.enable_punc:
            punc_results = []
            batch_asr_text, batch_asr_uttid, batch_asr_timestamp = [], [], []
            for j, asr_result in enumerate(asr_results):
                batch_asr_text.append(asr_result["text"])
                batch_asr_uttid.append(asr_result["uttid"])
                if self.config.asr_config.return_timestamp:
                    batch_asr_timestamp.append(asr_result.get("timestamp", []))
                elif "timestamp" in asr_result:
                    batch_asr_timestamp.append(asr_result["timestamp"])
                if len(batch_asr_text) < self.config.punc_batch_size \
                        and j != len(asr_results) - 1:
                    continue
                if with_ts:
                    batch_punc_results = self.punc.process_with_timestamp(
                        batch_asr_timestamp, batch_asr_uttid)
                else:
                    batch_punc_results = self.punc.process(batch_asr_text,
                                                           batch_asr_uttid)
                logger.info(f"Punc: {batch_punc_results}")
                punc_results.extend(batch_punc_results)
                batch_asr_text, batch_asr_uttid, batch_asr_timestamp = [], [], []
        else:
            punc_results = asr_results

        # 6. assemble
        sentences, words = [], []
        for asr_result, punc_result, lid_result in zip(asr_results, punc_results,
                                                       lid_results):
            assert asr_result["uttid"] == punc_result["uttid"], \
                f"fix code: {asr_result} | {punc_result}"
            start_ms, end_ms = asr_result["uttid"].split("_")[-2:]
            start_ms, end_ms = int(start_ms[1:]), int(end_ms[1:])
            if with_ts:
                sub_sentences = []
                if self.config.enable_punc:
                    for i, punc_sent in enumerate(punc_result["punc_sentences"]):
                        start = start_ms + int(punc_sent["start_s"] * 1000)
                        end = start_ms + int(punc_sent["end_s"] * 1000)
                        if i == 0:
                            start = start_ms
                        if i == len(punc_result["punc_sentences"]) - 1:
                            end = end_ms
                        sub = {"start_ms": start, "end_ms": end,
                               "text": punc_sent["punc_text"],
                               "asr_confidence": asr_result["confidence"],
                               "lang": None, "lang_confidence": 0}
                        if lid_result:
                            sub["lang"] = lid_result["lang"]
                            sub["lang_confidence"] = lid_result["confidence"]
                        sub_sentences.append(sub)
                else:
                    sub = {"start_ms": start_ms, "end_ms": end_ms,
                           "text": asr_result["text"],
                           "asr_confidence": asr_result["confidence"],
                           "lang": None, "lang_confidence": 0}
                    if lid_result:
                        sub["lang"] = lid_result["lang"]
                        sub["lang_confidence"] = lid_result["confidence"]
                    sub_sentences = [sub]
                sentences.extend(sub_sentences)
            else:
                text = (punc_result["punc_text"] if self.config.enable_punc
                        else asr_result["text"])
                sentence = {"start_ms": start_ms, "end_ms": end_ms, "text": text,
                            "asr_confidence": asr_result["confidence"],
                            "lang": None, "lang_confidence": 0}
                if lid_result:
                    sentence["lang"] = lid_result["lang"]
                    sentence["lang_confidence"] = lid_result["confidence"]
                sentences.append(sentence)

            if "timestamp" in asr_result:
                for w, s, e in asr_result["timestamp"]:
                    words.append({"start_ms": int(s * 1000 + start_ms),
                                  "end_ms": int(e * 1000 + start_ms), "text": w})

        vad_segments_ms = [(int(s * 1000), int(e * 1000))
                           for s, e in vad_result["timestamps"]]
        text = "".join(s["text"] for s in sentences)
        text = re.sub(r'([.,!?])\s*([a-zA-Z])', r'\1 \2', text)

        return {"uttid": uttid, "text": text, "sentences": sentences,
                "vad_segments_ms": vad_segments_ms, "dur_s": dur,
                "words": words, "wav_path": wav_path}
