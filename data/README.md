Datasets

This repo expects simple, transparent formats for quick experiments:

1) Instruction JSONL
   {"instruction": "...", "input": "optional context", "output": "gold answer"}

2) Chat JSONL
   {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}

Samples live under data/samples for smoke testing and CI sanity checks.

Tips
- Keep samples tiny and representative. Filter out extreme lengths.
- Track data revisions by saving frozen snapshots alongside experiments.
