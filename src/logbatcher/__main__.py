import logging
import sys
import os

from logbatcher.parser import LogBatcher


def main(prog, *argv):
    logging.basicConfig(level=logging.WARN)

    model = os.environ.get("LOGBATCHER_MODEL", "gpt-4o-mini")

    parser = LogBatcher(model=model)

    logs = [*sys.stdin]

    out = parser.parse(logs)

    for o in out:
        print(o)


if __name__ == "__main__":
    main(*sys.argv)
