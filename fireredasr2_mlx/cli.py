#!/usr/bin/env python3
"""fireredasr2s-mlx-cli: zero-torch four-module ASR system CLI.

Mirror of fireredasr2s_cli.py with the same arguments, defaults and outputs
(result.jsonl + TextGrid + SRT + optional segment dumps).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os

import soundfile as sf
from textgrid import IntervalTier, TextGrid

from . import FireRedAsr2MlxConfig
from .llm import FireRedAsr2LlmMlxConfig
from .models.lid import FireRedLidMlxConfig
from .models.vad import FireRedVadMlxConfig
from .punc.punc_mlx import FireRedPuncMlxConfig
from .system import FireRedAsr2SystemMlx, FireRedAsr2SystemMlxConfig

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
logger = logging.getLogger("fireredasr2s.asr_system")

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="FireRedASR2S four-module system, native MLX (Apple Silicon)")
input_g = parser.add_argument_group("Input Options")
input_g.add_argument("--wav_path", type=str)
input_g.add_argument("--wav_paths", type=str, nargs="*")
input_g.add_argument("--wav_dir", type=str)
input_g.add_argument("--wav_scp", type=str)
input_g.add_argument("--sort_wav_by_dur", type=int, default=0)

output_g = parser.add_argument_group("Output Options")
output_g.add_argument("--outdir", type=str, default="output")
output_g.add_argument("--write_textgrid", type=int, default=1)
output_g.add_argument("--write_srt", type=int, default=1)
output_g.add_argument("--save_segment", type=int, default=0)

module_g = parser.add_argument_group("Module Switches")
module_g.add_argument("--enable_vad", type=int, default=1, choices=[0, 1])
module_g.add_argument("--enable_lid", type=int, default=1, choices=[0, 1])
module_g.add_argument("--enable_punc", type=int, default=1, choices=[0, 1])

asr_g = parser.add_argument_group("ASR Options")
asr_g.add_argument("--asr_type", type=str, default="aed", choices=["aed", "llm"])
asr_g.add_argument("--asr_model_dir", type=str,
                   default="pretrained_models/FireRedASR2-AED")
asr_g.add_argument("--asr_use_gpu", type=int, default=1)
asr_g.add_argument("--asr_use_half", type=int, default=0)
asr_g.add_argument("--asr_batch_size", type=int, default=1)
# FireRedASR-AED
asr_g.add_argument("--beam_size", type=int, default=3)
asr_g.add_argument("--decode_max_len", type=int, default=0)
asr_g.add_argument("--nbest", type=int, default=1)
asr_g.add_argument("--softmax_smoothing", type=float, default=1.25)
asr_g.add_argument("--aed_length_penalty", type=float, default=0.6)
asr_g.add_argument("--eos_penalty", type=float, default=1.0)
asr_g.add_argument("--return_timestamp", type=int, default=1)
# FireRedASR-AED External LM
asr_g.add_argument("--elm_dir", type=str, default="")
asr_g.add_argument("--elm_weight", type=float, default=0.0)

vad_g = parser.add_argument_group("VAD Options")
vad_g.add_argument("--vad_model_dir", type=str,
                   default="pretrained_models/FireRedVAD/VAD")
vad_g.add_argument("--vad_use_gpu", type=int, default=1)
vad_g.add_argument("--vad_chunk_max_frame", type=int, default=30000)
vad_g.add_argument("--smooth_window_size", type=int, default=5)
vad_g.add_argument("--speech_threshold", type=float, default=0.2)
vad_g.add_argument("--min_speech_frame", type=int, default=20)
vad_g.add_argument("--max_speech_frame", type=int, default=1000)
vad_g.add_argument("--min_silence_frame", type=int, default=10)
vad_g.add_argument("--merge_silence_frame", type=int, default=50)
vad_g.add_argument("--extend_speech_frame", type=int, default=10)

lid_g = parser.add_argument_group("LID Options")
lid_g.add_argument("--lid_model_dir", type=str,
                   default="pretrained_models/FireRedLID")
lid_g.add_argument("--lid_use_gpu", type=int, default=1)

punc_g = parser.add_argument_group("Punc Options")
punc_g.add_argument("--punc_model_dir", type=str,
                    default="pretrained_models/FireRedPunc")
punc_g.add_argument("--punc_use_gpu", type=int, default=1)
punc_g.add_argument("--punc_batch_size", type=int, default=1)
punc_g.add_argument("--punc_with_timestamp", type=int, default=1)
punc_g.add_argument("--punc_sentence_max_length", type=int, default=-1)


def main(args):
    if args.elm_dir:
        logger.warning(
            f"External LM is not supported by the MLX port, ignoring "
            f"--elm_dir {args.elm_dir} --elm_weight {args.elm_weight} "
            f"(AED decoding runs without the external LM).")
    if args.asr_type == "llm" and args.return_timestamp:
        logger.info("--asr_type llm: the LLM variant has no word timestamps, "
                    "--return_timestamp has no effect.")
    # MLX runs on Metal/CPU; --asr_use_gpu/--asr_use_half/--vad_use_gpu/
    # --lid_use_gpu/--punc_use_gpu are accepted for CLI compatibility only.

    wavs = get_wav_info(args)
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
    fout = open(args.outdir + "/result.jsonl", "w") if args.outdir else None

    vad_config = FireRedVadMlxConfig(
        smooth_window_size=args.smooth_window_size,
        speech_threshold=args.speech_threshold,
        min_speech_frame=args.min_speech_frame,
        max_speech_frame=args.max_speech_frame,
        min_silence_frame=args.min_silence_frame,
        merge_silence_frame=args.merge_silence_frame,
        extend_speech_frame=args.extend_speech_frame,
        chunk_max_frame=args.vad_chunk_max_frame)
    lid_config = FireRedLidMlxConfig()
    asr_config = FireRedAsr2MlxConfig(
        beam_size=args.beam_size, nbest=args.nbest,
        decode_max_len=args.decode_max_len,
        softmax_smoothing=args.softmax_smoothing,
        aed_length_penalty=args.aed_length_penalty,
        eos_penalty=args.eos_penalty,
        return_timestamp=bool(args.return_timestamp))
    llm_config = FireRedAsr2LlmMlxConfig(
        beam_size=args.beam_size, decode_max_len=args.decode_max_len)
    punc_config = FireRedPuncMlxConfig(
        sentence_max_length=args.punc_sentence_max_length,
        punc_with_timestamp=args.punc_with_timestamp)

    system_config = FireRedAsr2SystemMlxConfig(
        vad_model_dir=args.vad_model_dir, lid_model_dir=args.lid_model_dir,
        asr_model_dir=args.asr_model_dir, punc_model_dir=args.punc_model_dir,
        asr_type=args.asr_type,
        vad_config=vad_config, lid_config=lid_config, asr_config=asr_config,
        llm_config=llm_config,
        punc_config=punc_config, asr_batch_size=args.asr_batch_size,
        punc_batch_size=args.punc_batch_size, enable_vad=args.enable_vad,
        enable_lid=args.enable_lid, enable_punc=args.enable_punc)
    asr_system = FireRedAsr2SystemMlx(system_config)

    for uttid, wav_path in wavs:
        logger.info("")
        result = asr_system.process(wav_path, uttid)
        logger.info(f"FINAL: {result}")
        if fout:
            fout.write(f"{json.dumps(result, ensure_ascii=False)}\n")
            fout.flush()
        name = os.path.basename(wav_path).replace(".wav", "")
        if args.write_textgrid:
            tg_dir = os.path.join(args.outdir, "asr_tg")
            write_textgrid(tg_dir, name, result["dur_s"], result["sentences"],
                           result["words"])
        if args.write_srt:
            srt_dir = os.path.join(args.outdir, "asr_srt")
            write_srt(srt_dir, name, result["sentences"])
        if args.save_segment:
            save_dir = os.path.join(args.outdir, "vad_segment")
            split_and_save_segment(wav_path, result["vad_segments_ms"], save_dir)

    if fout:
        fout.close()
    logger.info("All Done")


def get_wav_info(args):
    def base(p):
        return os.path.basename(p).replace(".wav", "")
    if args.wav_path:
        wavs = [(base(args.wav_path), args.wav_path)]
    elif args.wav_paths and len(args.wav_paths) >= 1:
        wavs = [(base(p), p) for p in sorted(args.wav_paths)]
    elif args.wav_scp:
        wavs = [line.strip().split() for line in open(args.wav_scp)]
    elif args.wav_dir:
        wavs = glob.glob(f"{args.wav_dir}/**/*.wav", recursive=True)
        wavs = [(base(p), p) for p in sorted(wavs)]
    else:
        raise ValueError("Please provide valid wav info")
    if getattr(args, "sort_wav_by_dur", 0):
        logger.info("Sort wav by duration...")

        def dur_s(p):
            info = sf.info(p)
            return info.frames / info.samplerate

        # Longest first, same order as the official sort (key = -dur).
        wavs = sorted(wavs, key=lambda x: -dur_s(x[1]))
        logger.info("Sort Done")
    logger.info(f"#wavs={len(wavs)}")
    return wavs


def write_textgrid(tg_dir, name, wav_dur, sentences, words=None):
    os.makedirs(tg_dir, exist_ok=True)
    textgrid_file = os.path.join(tg_dir, name + ".TextGrid")
    textgrid = TextGrid(maxTime=wav_dur)
    tier = IntervalTier(name="sentence", maxTime=wav_dur)
    for sentence in sentences:
        start_s = sentence["start_ms"] / 1000.0
        end_s = sentence["end_ms"] / 1000.0
        text = sentence["text"]
        confi = sentence["asr_confidence"]
        if start_s == end_s:
            continue
        start_s = max(start_s, 0)
        end_s = min(end_s, wav_dur)
        tier.add(minTime=start_s, maxTime=end_s, mark=f"{text}\n{confi}")
    textgrid.append(tier)
    if words:
        tier = IntervalTier(name="token", maxTime=wav_dur)
        for word in words:
            start_s = word["start_ms"] / 1000.0
            end_s = word["end_ms"] / 1000.0
            if start_s == end_s:
                continue
            tier.add(minTime=max(start_s, 0), maxTime=min(end_s, wav_dur),
                     mark=word["text"])
        textgrid.append(tier)
    textgrid.write(textgrid_file)


def write_srt(srt_dir, name, sentences):
    def _ms2srt_time(ms):
        h = ms // 1000 // 3600
        m = (ms // 1000 % 3600) // 60
        s = (ms // 1000 % 3600) % 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms % 1000:03d}"

    os.makedirs(srt_dir, exist_ok=True)
    srt_file = os.path.join(srt_dir, name + ".srt")
    with open(srt_file, "w") as fout:
        i = 0
        n = len(sentences)
        for sentence in sentences:
            if sentence["text"].strip() == "":
                continue
            i += 1
            fout.write(f"{i}\n")
            fout.write(f"{_ms2srt_time(sentence['start_ms'])} --> "
                       f"{_ms2srt_time(sentence['end_ms'])}\n")
            fout.write(f"{sentence['text']}\n")
            if i != n:
                fout.write("\n")


def split_and_save_segment(wav_path, timestamps_ms, save_segment_dir):
    os.makedirs(save_segment_dir, exist_ok=True)
    wav_np, sample_rate = sf.read(wav_path, dtype="int16")
    for i, (start_ms, end_ms) in enumerate(timestamps_ms):
        uttid = wav_path.split("/")[-1].replace(".wav", "")
        seg_path = f"{save_segment_dir}/{uttid}_{i}_{start_ms}_{end_ms}.wav"
        start = int(start_ms / 1000 * sample_rate)
        end = int(end_ms / 1000 * sample_rate)
        sf.write(seg_path, wav_np[start:end], samplerate=sample_rate)


def cli_main():
    args = parser.parse_args()
    logger.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
