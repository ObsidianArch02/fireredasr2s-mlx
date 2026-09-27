#!/usr/bin/env python3
"""One-time golden dump of the official FireRedASR2-LLM (PyTorch CPU, fp32).

Runs the official model on the 3 golden wavs and records every intermediate
an MLX parity test needs:
  feats / encoder_out / speech_features (adapter out) / merged embeddings /
  first-step logits / generated ids (beam search) / decoded text.

Outputs:
  golden/golden_llm.npz   — numpy arrays (torch-free to read)
  golden/golden_llm.json  — texts, ids, shapes

Usage (torch venv, official checkout in ../FireRedASR2S-main or
FIRERED_OFFICIAL_DIR):
  .venv/bin/python scripts/dump_golden_llm.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DIR = Path(os.environ.get(
    "FIRERED_OFFICIAL_DIR",
    PROJECT_ROOT.parent / "FireRedASR2S-main"))
LLM_DIR = PROJECT_ROOT / "pretrained_models" / "FireRedASR2-LLM"
WAVS = ["hello_zh.wav", "hello_en.wav", "event.wav"]

sys.path.insert(0, str(OFFICIAL_DIR))

from fireredasr2s.fireredasr2.asr import load_firered_llm_model_and_tokenizer
from fireredasr2s.fireredasr2.data.asr_feat import ASRFeatExtractor
from fireredasr2s.fireredasr2.tokenizer.llm_tokenizer import \
    LlmTokenizerWrapper


def main():
    torch.set_num_threads(os.cpu_count() or 8)

    model_path = str(LLM_DIR / "model.pth.tar")
    encoder_path = str(LLM_DIR / "asr_encoder.pth.tar")
    llm_path = str(LLM_DIR / "Qwen2-7B-Instruct")
    cmvn_path = str(LLM_DIR / "cmvn.ark")

    t0 = time.time()
    model, tokenizer = load_firered_llm_model_and_tokenizer(
        model_path, encoder_path, llm_path)
    model = model.float().cpu().eval()
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    feat_extractor = ASRFeatExtractor(cmvn_path)
    input_ids, attention_mask, _, _ = LlmTokenizerWrapper.preprocess_texts(
        origin_texts=[""], tokenizer=tokenizer, max_len=128, decode=True)
    print("prompt ids:", input_ids.tolist())

    out = {}
    texts = {}
    for wav_name in WAVS:
        uttid = wav_name.replace(".wav", "")
        wav_path = str(PROJECT_ROOT / "golden" / "wavs" / wav_name)
        feats, lengths, durs, _, _ = feat_extractor([wav_path], [uttid])
        feats = feats.float()
        print(f"[{uttid}] feats {tuple(feats.shape)} lens {lengths.tolist()}",
              flush=True)

        with torch.no_grad():
            enc_out, enc_lengths, _ = model.encoder(feats, lengths)
            speech_features, speech_lens = model.encoder_projector(
                enc_out, enc_lengths)
            embs = model.llm.get_input_embeddings()(input_ids)
            merged, merged_mask, _ = \
                model._merge_input_ids_with_speech_features(
                    speech_features.to(embs.dtype), embs, input_ids,
                    attention_mask, speech_lens=speech_lens)
            first_logits = model.llm(inputs_embeds=merged,
                                     attention_mask=merged_mask).logits[
                0, -1].float()
            top_v, top_i = torch.topk(first_logits, 10)

            gen_ids = model.transcribe(
                feats, lengths, input_ids, attention_mask,
                beam_size=3, decode_max_len=0, decode_min_len=0,
                repetition_penalty=1.0, llm_length_penalty=0.0,
                temperature=1.0)

        text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        print(f"[{uttid}] gen ids: {gen_ids.tolist()}", flush=True)
        print(f"[{uttid}] text: {text}", flush=True)
        print(f"[{uttid}] first logits top10: {list(zip(top_i.tolist(), [round(v, 4) for v in top_v.tolist()]))}", flush=True)

        out[f"{uttid}__feats"] = feats.numpy().astype(np.float32)
        out[f"{uttid}__lengths"] = lengths.numpy().astype(np.int64)
        out[f"{uttid}__enc_out"] = enc_out.numpy().astype(np.float32)
        out[f"{uttid}__enc_lengths"] = enc_lengths.numpy().astype(np.int64)
        out[f"{uttid}__speech_features"] = \
            speech_features.numpy().astype(np.float32)
        out[f"{uttid}__speech_lens"] = speech_lens.numpy().astype(np.int64)
        out[f"{uttid}__merged"] = merged.numpy().astype(np.float32)
        out[f"{uttid}__merged_mask"] = merged_mask.numpy().astype(np.int64)
        out[f"{uttid}__first_logits"] = first_logits.numpy().astype(np.float32)
        texts[uttid] = {"ids": gen_ids[0].tolist(), "text": text[0]}

    np.savez(PROJECT_ROOT / "golden" / "golden_llm.npz", **out)
    (PROJECT_ROOT / "golden" / "golden_llm.json").write_text(
        json.dumps(texts, ensure_ascii=False, indent=2))
    print("saved golden/golden_llm.npz + golden_llm.json")


if __name__ == "__main__":
    main()
