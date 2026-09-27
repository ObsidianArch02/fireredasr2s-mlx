"""Torch-free LLM tokenizer wrapper.

Mirrors fireredasr2/tokenizer/llm_tokenizer.py (LlmTokenizerWrapper) on top of
the `tokenizers` runtime (Rust core, no torch/transformers): the Qwen2
tokenizer plus one added special token <speech>. Only the decode-side prompt
construction is ported (the 4-module system always transcribes from audio,
never from reference text).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tokenizers import AddedToken, Tokenizer

DEFAULT_SPEECH_TOKEN = "<speech>"


class LlmTokenizerWrapperMlx:
    @classmethod
    def build_llm_tokenizer(cls, llm_dir):
        llm_dir = Path(llm_dir)
        tok = Tokenizer.from_file(str(llm_dir / "tokenizer.json"))
        tok.add_tokens([AddedToken(DEFAULT_SPEECH_TOKEN, special=True)])
        # official asserts the configured pad token is <|endoftext|>
        cfg = json.loads((llm_dir / "tokenizer_config.json").read_text())
        assert cfg.get("pad_token") == "<|endoftext|>", \
            f"unexpected pad_token {cfg.get('pad_token')}"
        assert tok.token_to_id("<|endoftext|>") is not None
        return cls(tok)

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @property
    def pad_token_id(self):
        return self.tokenizer.token_to_id("<|endoftext|>")

    @property
    def eos_token_id(self):
        return self.tokenizer.token_to_id("<|im_end|>")

    @property
    def bos_token_id(self):
        return self.tokenizer.token_to_id("<|im_start|>")

    @property
    def speech_token_id(self):
        return self.tokenizer.token_to_id(DEFAULT_SPEECH_TOKEN)

    def decode_prompt(self):
        """Prompt for audio transcription (official decode=True template).

        Returns (input_ids (1, U) int64 np, attention_mask (1, U) int64 np).
        """
        text = (f"<|im_start|>user\n{DEFAULT_SPEECH_TOKEN}请转写音频为文字"
                f"<|im_end|>\n<|im_start|>assistant\n")
        ids = self.tokenizer.encode(text).ids
        input_ids = np.asarray([ids], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)
        return input_ids, attention_mask

    def batch_decode(self, batch_ids):
        return [self.tokenizer.decode(list(ids), skip_special_tokens=True)
                for ids in batch_ids]
