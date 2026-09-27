"""Pure-Python BertTokenizer equivalent (BasicTokenizer + WordPiece).

Reimplements HuggingFace BertTokenizer.tokenize(do_lower_case=True,
tokenize_chinese_chars=True, strip_accents implied by lowercasing) plus the
official _recover_unk post-pass, verbatim from
fireredpunc/data/hf_bert_tokenizer.py.
"""

from __future__ import annotations

import re
import unicodedata


def _is_whitespace(ch):
    if ch in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(ch) == "Zs"


def _is_control(ch):
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch).startswith("C")


def _is_chinese_char(cp):
    return ((0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF)
            or (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F))


def _is_punctuation(ch):
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")


class BasicTokenizer:
    def __init__(self, do_lower_case=True, tokenize_chinese_chars=True):
        self.do_lower_case = do_lower_case
        self.tokenize_chinese_chars = tokenize_chinese_chars

    def tokenize(self, text):
        text = self._clean_text(text)
        out = []
        for token in text.split():
            if self.tokenize_chinese_chars:
                token = self._tokenize_chinese_chars(token)
            for tok in token.split():
                if self.do_lower_case:
                    tok = tok.lower()
                    tok = self._run_strip_accents(tok)
                out.extend(self._run_split_on_punc(tok))
        return out

    def _clean_text(self, text):
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or _is_control(ch):
                continue
            out.append(" " if _is_whitespace(ch) else ch)
        return "".join(out)

    def _tokenize_chinese_chars(self, text):
        out = []
        for ch in text:
            if _is_chinese_char(ord(ch)):
                out.append(" " + ch + " ")
            else:
                out.append(ch)
        return "".join(out)

    def _run_strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    def _run_split_on_punc(self, text):
        chars = list(text)
        out, start_new = [], True
        for i, ch in enumerate(chars):
            if _is_punctuation(ch):
                out.append([ch])
                start_new = True
            else:
                if start_new:
                    out.append([])
                start_new = False
                out[-1].append(ch)
        return ["".join(t) for t in out]


class WordpieceTokenizer:
    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        out = []
        for token in text.split():
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                out.append(self.unk_token)
                continue
            is_bad, start, sub_tokens = False, 0, []
            while start < len(chars):
                end = len(chars)
                cur = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur = substr
                        break
                    end -= 1
                if cur is None:
                    is_bad = True
                    break
                sub_tokens.append(cur)
                start = end
            out.extend(sub_tokens if not is_bad else [self.unk_token])
        return out


class BertWordPieceTokenizer:
    """drop-in for the official HfBertTokenizer (zero deps)"""

    def __init__(self, vocab_path, do_lower_case=True):
        with open(vocab_path, encoding="utf-8") as f:
            self.vocab = {line.rstrip("\n"): i for i, line in enumerate(f)}
        self.basic = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece = WordpieceTokenizer(self.vocab)

    def tokenize(self, text, recover_unk=False):
        tokens = self.wordpiece.tokenize(" ".join(self.basic.tokenize(text)))
        tokens_id = [self.vocab.get(t, self.vocab["[UNK]"]) for t in tokens]
        if recover_unk and "[UNK]" in tokens:
            tokens = self._recover_unk(text.lower(), tokens)
        return tokens, tokens_id

    def _recover_unk(self, text, tokens):
        if "[UNK]" not in tokens:
            return tokens
        new_tokens = []
        text_no_space = text.replace(" ", "")
        if re.match(r"^[^a-zA-Z0-9']+$", text):
            tmp_text = text_no_space
            if len(tmp_text) == len(tokens):
                success = True
                for t, tok in zip(tmp_text, tokens):
                    if tok != "[UNK]" and t != tok:
                        success = False
                        break
                    new_tokens.append(t)
                if success:
                    return new_tokens
        new_tokens = []
        text_pos = 0
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "[UNK]":
                unk_count = 0
                j = i
                while j < len(tokens) and tokens[j] == "[UNK]":
                    unk_count += 1
                    j += 1
                post_token = ""
                if j < len(tokens):
                    post_token = tokens[j].replace("##", "")
                if post_token:
                    remaining = text_no_space[text_pos:]
                    anchor_pos = remaining.find(post_token)
                    unk_chars = (remaining[:anchor_pos] if anchor_pos != -1
                                 else remaining[:unk_count])
                else:
                    unk_chars = text_no_space[text_pos:text_pos + unk_count]
                for k in range(unk_count):
                    new_tokens.append(unk_chars[k] if k < len(unk_chars) else "")
                text_pos += len(unk_chars)
                i = j
            else:
                new_tokens.append(token)
                text_pos += len(token.replace("##", ""))
                i += 1
        return [t for t in new_tokens if t and t != "[UNK]"]
