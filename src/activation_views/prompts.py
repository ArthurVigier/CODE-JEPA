from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptRecord:
    prompt_id: str
    source: str
    category: str
    text: str


def stratified_phase0_prompts() -> list[PromptRecord]:
    templates = {
        "the_stack_python": "Write a Python function for task #{i} and explain edge cases.",
        "gsm8k": "Solve this arithmetic word problem #{i} step by step.",
        "big_bench_hard": "Reason about this logic puzzle #{i} carefully.",
        "c4_narrative": "Continue this narrative passage #{i} coherently.",
        "spider_sql": "Write the SQL query for database question #{i}.",
    }
    prompts: list[PromptRecord] = []
    for source, template in templates.items():
        for i in range(100):
            category = source
            prompts.append(
                PromptRecord(
                    prompt_id=f"{source}-{i:03d}",
                    source=source,
                    category=category,
                    text=template.format(i=i),
                )
            )
    return prompts
