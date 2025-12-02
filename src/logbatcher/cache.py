from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple

from logbatcher.template import Template

class TemplateNode(NamedTuple):
    event_id: int
    template: Template


@dataclass(eq=True, slots=True, init=False)
class TemplateTree:
    edges: dict[str, TemplateTree]
    leaf: TemplateNode | None

    def __init__(self):
        self.edges = defaultdict(TemplateTree)
        self.leaf = None

    def match(self, tokens: list[str]) -> tuple[TemplateNode, list[str]] | None:
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
        self.cache: dict[str, TemplateNode] = {}
        self.variable_candidates: set[str] = set()

    def insert(self, template: Template):
        tree = self.template_tree

        if len(template.tokens) == 0:
            raise ValueError(f"invalid {template=}")

        for token in template.tokens:
            tree = tree.edges[token]

        if tree.leaf is not None:
            return

        event_id = len(self.template_list)
        self.template_list.append(template)
        tree.leaf = TemplateNode(
            event_id,
            template,
        )

    def match(self, log: str) -> TemplateNode | None:
        tokens = Template.tokenize(log)
        if (result := self.template_tree.match(tokens)) is None:
            return None
        leaf, _ = result
        return leaf

    def __contains__(self, template: Template):
        tree = self.template_tree

        for token in template.tokens:
            if token not in tree.edges:
                return False
            tree = tree.edges[token]

        return True
