"""Fails if a relative link in a markdown file points to a path that does not exist."""
import re
import sys
from pathlib import Path

bad = []
for md in Path(".").rglob("*.md"):
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", md.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (md.parent / target).exists():
            bad.append("%s -> %s" % (md, target))
print("\n".join(bad) or "all relative links resolve")
sys.exit(1 if bad else 0)
