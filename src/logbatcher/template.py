import re
from dataclasses import dataclass
from typing import ClassVar, Self

WILDCARD = "<*>"


@dataclass(eq=True, frozen=True, slots=True)
class Template:
    tokens: list[str]

    token_pattern: ClassVar[re.Pattern] = re.compile(r"(?:\w|<\*>)+|[^\w\s]")

    @classmethod
    def from_str(cls, string: str) -> Self:
        return cls(cls.tokenize(string))

    @staticmethod
    def tokenize(string: str) -> list[str]:
        tokens = re.findall(r"(?:\w|<\*>)+|[^\w\s]", string)
        tokens = [WILDCARD if WILDCARD in tok else tok for tok in tokens]

        if len(tokens) < 2:
            return tokens

        out = tokens[:1]
        for token in tokens[1:]:
            if out[-1] == "<*>" and token == "<*>":
                continue
            out.append(token)

        return out

    def extract(self, string: str) -> list[str] | None:
        out = []
        tokens = self.tokenize(string)
        for part, token in zip(self.tokens, tokens):
            if part == WILDCARD:
                out.append(token)
                continue
            elif part != token:
                return None

        return out
