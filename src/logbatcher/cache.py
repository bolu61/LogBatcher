from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple

from logbatcher.tokenize import tokenize

type Template = list[str]


class TemplateTreeLeaf(NamedTuple):
    event_id: int
    template: Template


@dataclass(eq=True, slots=True, init=False)
class TemplateTree:
    edges: dict[str, TemplateTree]
    leaf: TemplateTreeLeaf | None

    def __init__(self):
        self.edges = defaultdict(TemplateTree)
        self.leaf = None

    def match(self, tokens: list[str]) -> tuple[TemplateTreeLeaf, list[str]] | None:
        if len(tokens) == 0:
            if self.leaf is not None:
                return self.leaf, []
            return None

        head, *rest = tokens
        if head in self.edges:
            if (result := self.edges[head].match(rest)) is not None:
                return result

        if "<*>" in self.edges:
            if (result := self.edges["<*>"].match(rest)) is not None:
                leaf, vars = result
                return leaf, vars + [head]

        return None


class ParsingCache(object):
    def __init__(self):
        self.template_tree: TemplateTree = TemplateTree()
        self.template_list: list[Template] = []
        self.cache: dict[str, TemplateTreeLeaf] = {}
        self.variable_candidates: set[str] = set()

    def insert(self, template: Template):
        tree = self.template_tree

        if len(template) == 0:
            raise ValueError(f"invalid {template=}")

        for token in template:
            tree = tree.edges[token]

        if tree.leaf is not None:
            return

        event_id = len(self.template_list)
        self.template_list.append(template)
        tree.leaf = TemplateTreeLeaf(
            event_id,
            template,
        )

    def match(self, log: str) -> TemplateTreeLeaf | None:
        tokens = tokenize(log)
        if (result := self.template_tree.match(tokens)) is None:
            return None
        leaf, _ = result
        return leaf

    def __contains__(self, template: Template):
        tree = self.template_tree

        for token in template:
            if token not in tree.edges:
                return False
            tree = tree.edges[token]

        return True
