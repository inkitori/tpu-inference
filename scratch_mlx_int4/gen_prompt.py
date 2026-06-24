#!/usr/bin/env python
"""Generate a ~4000-token prompt string and print it (single line, JSON-safe
not needed since we use --data-binary via a file)."""
import sys
n_words = int(sys.argv[1]) if len(sys.argv) > 1 else 3200
# ~1.25 tokens/word-ish; build a varied prompt to avoid trivial caching.
base = ("The history of computing spans many centuries and involves "
        "mathematicians engineers and scientists from around the world "
        "who contributed ideas about calculation logic and machinery ")
words = (base * 200).split()
out = " ".join(words[:n_words])
sys.stdout.write(out)
