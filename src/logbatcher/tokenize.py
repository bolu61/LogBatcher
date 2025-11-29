import re


def tokenize(message: str) -> list[str]:
    tokens = re.findall(r"(?:\w|<\*>)+|[^\w\s]", message)
    tokens = ["<*>" if "<*>" in tok else tok for tok in tokens]

    if len(tokens) < 2:
        return tokens

    out = tokens[:1]
    for token in tokens[1:]:
        if out[-1] == "<*>" and token == "<*>":
            continue
        out.append(token)

    return out
