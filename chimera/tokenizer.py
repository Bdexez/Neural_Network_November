class ByteTokenizer:
    """Tokenizer UTF-8 sans vocabulaire externe, réversible pour tout texte Unicode."""

    pad_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 3
    vocab_size = 260

    def encode(self, text: str, *, bos=True, eos=True) -> list[int]:
        return ([self.bos_id] if bos else []) + [b + 4 for b in text.encode("utf-8")] + (
            [self.eos_id] if eos else []
        )

    def decode(self, tokens) -> str:
        return bytes(int(t) - 4 for t in tokens if 4 <= int(t) < 260).decode(
            "utf-8", errors="replace"
        )
