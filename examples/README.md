# APOBench examples

Four self-contained walkthroughs, from "confirm your install works" to
"extend the framework with your own method/dataset."

| Example | Start here if you want to... |
|---|---|
| [`01_quickstart_single_run/`](01_quickstart_single_run/) | Run one optimizer on one task and see a result — the smallest working example. |
| [`02_multi_seed_sweep/`](02_multi_seed_sweep/) | Run the real methods × datasets × seeds protocol — what "APOBench" as a citable comparison actually means. |
| [`03_custom_optimizer/`](03_custom_optimizer/) | Add your own optimization method and register it with the framework. |
| [`04_custom_dataset/`](04_custom_dataset/) | Bring your own task/dataset, either as a plain JSON file or a registered loader. |

Each example is runnable as-is (small models, small sample sizes, tight
budget caps) and documents its own expected runtime/cost. See
[`docs/OUTPUT_SCHEMA.md`](../docs/OUTPUT_SCHEMA.md) for the result format
every example produces, and the main [`README.md`](../README.md) for the
full architecture and method list.
