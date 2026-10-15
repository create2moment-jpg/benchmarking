# Evaluation methodology

## What is measured

Engineering task outcomes under controlled conditions - not tokens/second, not training. Wall
time and provider-reported usage are recorded as secondary context.

## Scoring

Each task defines evaluator **checks**; each check maps to one **dimension** and has a weight.

| Dimension | Default weight | Typical evidence |
|---|---|---|
| functional | 45 | hidden tests / fixtures / SQL result sets / C++ CHECK lines; public tests as regression guards |
| regression | 15 | `tests_detect_bug` (candidate tests fail on the original code and pass on the fix), `added_tests` |
| quality | 15 | ruff on changed Python files, `-Wall -Wextra -Werror` syntax checks for C++ |
| robustness | 10 | separate hidden tests for malformed input, boundaries, NaN, invalid parameters |
| workflow | 10 | from the harness trace: finished cleanly, tool-call success rate, expected artifacts, malformed replies |
| explanation | 5 | heuristic summary check (present, concise, mentions verification and limitations) or rubric |

* A check yields a score in [0, 1] (partial credit where meaningful, e.g. passed/collected tests).
* A dimension score is the weighted mean of its *scored* checks. Checks with `error` (evaluator
  infrastructure problem) or `skipped` are excluded and listed in notes - never counted as a
  model failure, never counted as a pass.
* The **aggregate** is 0-100 over *assessed* dimensions; **coverage** reports the share of total
  weight that was assessed. Compare aggregates only at similar coverage. Per-dimension scores are
  primary; the aggregate is secondary.
* **passed** = every scored functional check passed. Pass rate is the headline correctness metric.
* No-change attempts receive no quality credit (linting untouched fixture code would reward doing
  nothing).
* Task-specific weights override defaults (e.g. SQL tasks have no quality dimension).

### Score sources
`deterministic` (executed evaluators), `heuristic` (summary check), `human` (blind reviews),
`judge` (judge model with recorded identity and rationale). Human and judge scores are stored in
`reviews` and reported separately with inter-reviewer disagreement; they never silently replace
deterministic scores.

## Rubric and human-review tasks

`code_review.*` and `reasoning.*` tasks use deterministic text rubrics (explicit regex
properties with evidence of which criteria matched) as a reproducible *proxy*, and are marked
`evaluation: rubric | human_review` in the catalogue. For decisions, run blind human review:

```
bench review export --run-id RUN_A,RUN_B --out review/ --dimension functional,explanation
# reviewers fill review/reviews_template.csv using only review/packet.json (no model identity)
bench review import --key review/key.json --file filled.csv
```

Two or more reviewers per item give a disagreement value (max - min); > 0.3 is flagged.

## Fairness and comparability

* Identical task contract, acceptance criteria and tool set for every model in a harness.
* The built-in agent harness uses a provider-neutral JSON action protocol, so local and paid
  models are driven identically. Models tuned for native tool-calling APIs may underperform it;
  compare harnesses as their own dimension (`--harness direct|agent|<cli>`).
* Runs are comparable when task hash, environment fingerprint and configuration fingerprint
  match. Reports and the dashboard flag differing task versions/hashes and environments.
* Sampling parameters, quantization, serving runtime and context limit are recorded.
* Skipped tasks (missing optional runtime) are reported as skipped and excluded from rates.

## Statistics

Per (model, harness) group: pass rate, score mean / median / sd / min / max, median and p90
agent wall time, timeout rate, failure rate and failure classes, tool-call success rate,
regression rate (share of attempts failing a regression-guard check), malformed outputs, and
usage/cost **only** from attempts that reported them (shown as n of N). Groups with fewer than 5
scored attempts carry a small-sample warning; mock-provider groups carry a mock warning.

Use repeats (`--repeats N`, extended profile default 2) only where outputs are nondeterministic
(sampling temperature > 0, remote models); deterministic local runs at temperature 0 with a
seed rarely need them.

## Limitations and interpretation

* Small starter catalogue: 14 tasks. Differences of a few points on a handful of tasks are noise.
* Hidden tests bound - but do not prove - correctness.
* Heuristic explanation scoring is crude; weight it low (default 5) or replace with reviews.
* The fake provider replays reference solutions: its 100% scores validate the *pipeline*.
* Wall time depends on hardware and provider load; compare it only within one environment.
* Prompt contamination: public benchmarks can leak into training data; keep private variants of
  tasks for decisions that matter, and bump task versions when changing them.
