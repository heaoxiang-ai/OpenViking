# Agent Memory Leaderboard adapter

This directory contains three deliberately separated stages:

1. `server.py` exposes the competition-facing AML Add/Search API;
2. `eval/run_public.py` reads the already prepared datasets under
   `result/aml_data` and produces retrieval results;
3. `eval/evaluate.py` optionally runs each dataset's public Answer/Judge pipeline
   after retrieval, including deterministic PersonaMem MCQ grading.

The deployed adapter never imports the local evaluation code. In a formal AML
run, the platform supplies the datasets and controls Answer, Judge, and score
aggregation.

## Add/Search adapter

For a local OpenViking server using `server.auth_mode: dev`:

```bash
export OPENVIKING_URL=http://127.0.0.1:1933
export AML_API_KEY='<memory-system key>'

python -m benchmark.aml.server --host 127.0.0.1 --port 8088
```

Do not set an OpenViking key or account for this local `dev` setup. The SDK
passes each AML `user_id` directly as the OpenViking user; there is no hash or
second identity mapping. Like other OpenViking SDK clients, it also reads an
existing `~/.openviking/ovcli.conf`, so that file must match the server being
used (or be disabled with `OPENVIKING_CLI_CONFIG_FILE` pointing to a missing
file).

For a remote multi-user deployment, configure the OpenViking server with
`server.auth_mode: trusted`, then set the trusted-mode root key and account:

```bash
export OPENVIKING_URL='https://openviking.example.com'
export OPENVIKING_API_KEY='<trusted-mode root key>'
export OPENVIKING_ACCOUNT=default
export AML_API_KEY='<memory-system key>'

python -m benchmark.aml.server --host 0.0.0.0 --port 8088
```

OpenViking's standard `api_key` mode cannot use one root key to switch among
dynamic AML users: in that mode a request must use the key belonging to the
target user. `AML_API_KEY` is unrelated; it only protects this adapter's
incoming `/add` and `/search` endpoints.

The adapter provides:

- `GET /health`
- `POST /add`
- `POST /search`

`/add` calls the OpenViking SDK's `batch_add_messages`, `commit_session`, and
`get_task` in that order. `/search` calls OpenViking `find`, restricted to
`viking://~/memories` and level 2 under the same AML user. Each dataset case is
isolated by its `user_id`; session IDs only separate histories inside that
user's space. Transient SDK calls are retried up to three times by
default; use `AML_RETRY_ATTEMPTS` and `AML_RETRY_DELAY_SECONDS` to adjust that
behavior. `Token`, `Bearer`, and `X-Api-Key` authentication are accepted when
`AML_API_KEY` is set.

`/add` accepts either a content string (Textual/Coding) or an ordered array of
`text` and `image_url` parts in `messages[].content` (Multimodal). Strings are
forwarded as native `content`; arrays are forwarded as native Session `parts`,
preserving part order, message roles, and timestamps. Images must be inline
Base64 JPEG, PNG, or WebP Data URIs, with at most 10 MiB decoded per image and
30 MiB decoded across one Add request. For example:

```json
{
  "request_id": "example:add:1",
  "user_id": "example:user",
  "session_id": "example:session",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Remember this receipt."},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,<image-bytes>"}}
    ]
  }]
}
```

The example's image payload is a placeholder; supply actual Base64 image bytes.
Image understanding and memory extraction use OpenViking's existing Session
commit pipeline and require a configured vision-capable VLM. The adapter does
not perform separate image captioning.

`/search` accepts a query string or an array of the same `text` and `image_url`
parts, including image-only and multiple-image queries. Text parts are joined
with newlines and image URLs retain their relative order. The adapter forwards
them to native `find` using `query` and `image_url`, without an extra VLM call.
String queries keep their original request and response behavior. For example:

```json
{
  "user_id": "example:user",
  "query": [
    {"type": "text", "text": "Which workshop was this receipt for?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<image-bytes>"}}
  ],
  "top_k": 5
}
```

The same 10 MiB per-image limit applies, with a 30 MiB decoded total per Search
request. Image queries require a multimodal dense embedding model compatible
with the stored vectors. In hybrid mode, OV embeds the complete image/text
input through dense and only the text through sparse; image-only queries skip
sparse. Volcengine text-only queries retain their combined dense/sparse request.
Keep the configured dense model and dimension consistent with the existing
index. Search responses remain text memory evidence rather than raw images.

The native `find` call uses quick retrieval without reranking, even when the
OpenViking server has a reranker configured. The adapter returns each hit's
`content` when present, otherwise its `abstract`, without an additional
full-text read.

Run these modules from a source checkout with the project's Python dependencies
installed. The standard OpenViking image and installed package do not include
`benchmark/aml`.

## Local Add/Search

The input files already exist locally and are not generated by this code:

```bash
python -m benchmark.aml.eval.run_public \
  --prepared result/aml_data/prepared/lme_locomo_script.json \
  --prepared result/aml_data/prepared/persona_cl.json \
  --prepared result/aml_data/prepared/beam.json \
  --namespace local-check-v1 \
  --base-url http://127.0.0.1:8088 \
  --output-dir result/aml-public/retrieval \
  --add-concurrency 32 --search-concurrency 16
```

The main flow stops after Add/Search and writes one `retrieval.jsonl` under
each runnable dataset directory, plus a small top-level `summary.json`. Empty
private-data declarations are ignored because they contain nothing that can be
sent to Add or Search.

The prepared files contain `history_key`, not an AML `user_id`. The local
runner therefore creates one stable AML user for each unique
`namespace + unit + history_key + history contents`: questions sharing a
history share that user, while different histories stay isolated. This is ID
creation in the local runner; `server.py` forwards the resulting ID unchanged.

Use repeated `--unit` arguments to select individual datasets. Add runs up to
16 histories (AML users) concurrently by default, while Search runs up to 16 questions
concurrently. Override them independently with `--add-concurrency` and
`--search-concurrency`; the existing `--concurrency` option remains a shorthand
baseline for both. Phase-specific options take precedence. Within each history,
sessions run in prepared order, and chunks within each session run in order.
Every Add waits for its commit to finish before the next request for that user;
Search starts after all histories finish importing. For BEAM, this preserves
batch order within a conversation while different conversations run concurrently.

For smoke/full presets, `run_ecs` accepts a provider URL through `--base-url`
or `AML_BASE_URL`; it defaults to the local adapter. It reads authentication
from `AML_API_KEY` or `--api-key-file` and creates a fresh namespace per run:

```bash
python -m benchmark.aml.eval.run_ecs list
python -m benchmark.aml.eval.run_ecs full beam_100k \
  --base-url http://127.0.0.1:8088 --add-concurrency 32 --dry-run
```

Remove `--dry-run` to import. The generic callers do not resume an interrupted
import. Reconcile the original server tasks and archives before retrying an
ambiguous Add; a new namespace starts a separate experiment.

## PersonaMem-v2 text preparation

`eval/prepare_personamem.py` follows the exact `chat_history_32k_link` and
`chat_history_128k_link` paths in `benchmark/text/benchmark.csv`. Each size is a
separate condition; their histories are not concatenated. Shared histories
are imported once per run and reused by their questions.

The preparer removes all system persona messages and blank messages, preserves
the remaining user/assistant text and order, and omits timestamps absent from
the source. It does not assemble histories from raw persona records or gold
snippets. Reference answers, preference annotations and distractors stay in
case metadata and never enter Add messages.

```bash
python -m benchmark.aml.eval.prepare_personamem \
  --source-dir result/aml_data/datasets/personamem_v2 \
  --verification result/aml_data/personamem_v2_download/verification.json \
  --output-dir result/aml_data/prepared/full
```

The downloaded source and its verification manifest must already exist. The
manifest contains `success: true`, the source `revision`, and a `files` array
with relative `path` and `sha256` for the CSV and linked chat files. Preparation
checks those hashes, writes both size-specific prepared files and backs up any
replaced outputs. Optional `--smoke-file` refreshes existing 32K smoke cases.
Preparation makes no Add/Search or model calls and covers text inputs only.

## Memory-type contribution (LoCoMo-Refined and LongMemEval-S)

`eval/memory_contribution.py` measures how Event, Preference, Entity, and Profile
affect final Answer/Judge correctness. It uses one frozen retrieval artifact
and all 16 subsets of those four types. Removing a type only deletes its entries
from `retrieved_context`: there is no new Add, Search, rerank, link traversal,
backfill, or change to the remaining memory text/order. Supply the context that
Answer actually consumed, including any owner labels and original truncation.
The all-types condition must reproduce that context, not a larger candidate pool.

Each input JSONL row needs `id`, `unit`, `question`, a gold-answer field accepted
by the public pipeline, and `retrieved_context` (a list of strings). It also
needs `user_id` or `history_key` for history-cluster confidence intervals; use
`--group-field` for a differently named history identifier. Preserve question
types/categories and other original question metadata. Memory types require
provenance, using aligned `retrieved_uris`, aligned `retrieved_memory_types`, or
`MEMORY_FIELDS.memory_type` in the original text. Conflicting hints fail validation.
The public retrieval runner now retains user/history IDs and returned memory URIs.
For older text-only artifacts, attach aligned types/URIs from their frozen
retrieval receipts; do not guess the type from what the prose discusses.

Singular/plural Event, Preference, and Entity labels are normalized. Other
declared memory types (for example Identity and Soul) remain in every subset
and their entry count is disclosed. Thus subset 0 means **without these four
types**, not necessarily without any memory. A story or ChatLog embedded in an
Event belongs to Event; facts repeated in another type remain there. Inputs with
`speaker_1_memories`, `speaker_2_memories`, `memories`, or `chat_history` are rejected
because those fields could bypass the deletion; first represent the exact Answer
memory as ordered `retrieved_context` entries with provenance.

First prepare and inspect a model-call-free plan:

```bash
python -m benchmark.aml.eval.memory_contribution prepare \
  --unit longmemeval_s \
  --retrieval result/aml-public/retrieval/longmemeval_s/retrieval.jsonl \
  --aml-repo result/aml-public-eval/agent-memory-leaderboard \
  --output-dir result/aml-public/contribution/longmemeval_s
```

Repeat with `--unit locomo_refined` and its retrieval artifact in a separate
output directory. The default is five rounds, GPT-4o mini
(`gpt-4o-mini-2024-07-18`) for both stages, temperature 0, and max_tokens 16384.
`--answer-model`/`--judge-model` (also defaulted from `ANSWER_MODEL`/`JUDGE_MODEL`),
`--rounds`, `--temperature`, and `--max-tokens` are frozen at preparation.
The pinned public pipeline owns prompt rendering and Judge-label parsing;
this runner sends the explicitly frozen model/temperature/output limit rather
than inheriting the public pipeline's request defaults. Its default AML commit
matches `evaluate.py`; `--aml-commit` selects another verified clean checkout.
Existing historical runs are comparable only when their complete conditions match.
For N questions and R rounds, the plan requires 16 × N × R Answer requests
and as many Judge requests, before technical retries. Five rounds for the
861-question LoCoMo-R and 500-question LME-S sets total 217,760 requests.

After setting `ANSWER_API_BASE`, `ANSWER_API_KEY`, `JUDGE_API_BASE`, and
`JUDGE_API_KEY`, run or resume that exact plan:

```bash
python -m benchmark.aml.eval.memory_contribution run \
  --aml-repo result/aml-public-eval/agent-memory-leaderboard \
  --output-dir result/aml-public/contribution/longmemeval_s \
  --answer-concurrency 48 --judge-concurrency 48
```

Each round/subset has separate receipts containing the complete request,
provider response/usage, request SHA, and a checked artifact hash. API keys and
Authorization headers are not written. No results are reused across rounds or
subsets. Resume reuses the same job's first valid response; even a valid wrong
answer is retained. HTTP 408/429/500/502/503/504 and transport failures have at
most three attempts, counted across resumes, with shared stage cooldown and
Retry-After handling. Every attempt is preserved. Empty, truncated, or malformed
successful responses stop the run and are not resampled on resume. Missing or
invalid judgments prevent reporting; they never become a WRONG score.
The frozen input/script/pipeline hashes and runtime endpoints/settings must
match on resume. A `.running` lock prevents overlapping commands; after a hard
process kill, confirm the recorded PID has exited before removing that lock.

The run automatically reports once every Answer/Judge cell exists. Reporting
can also be repeated offline with a fixed bootstrap seed:

```bash
python -m benchmark.aml.eval.memory_contribution report \
  --output-dir result/aml-public/contribution/longmemeval_s \
  --bootstrap-samples 1000 --seed 0
```

Outputs are `summary.json`, `summary.md`, and `per-question.json`, in addition
to `manifest.json`, the single frozen `cases.json` snapshot, runtime settings,
and per-request receipts. Subset masks use bits 0–3 in the order Event,
Preference, Entity, Profile. Reports retain all 16 correctness values per
question/round and provide overall and question-type/category aggregates:

- **Shapley contribution (pp):** average marginal accuracy gain over all 24
  orders of adding the four types, calculated exactly from the 16 subsets.
  Redundant and complementary evidence share credit; contributions may be
  negative. Their sum equals all-types accuracy minus subset-0 accuracy.
- **Removal loss (pp):** all-types accuracy minus accuracy without that type.
- **Standalone gain (pp):** that type alone minus subset-0 accuracy.

All three include paired history-cluster/round percentile bootstrap 95% intervals.
Histories are sampled as groups while retaining their question counts; round
indices are shared across conditions. Few independent histories or rounds limit
the uncertainty estimate. LoCoMo-R and LME-S remain separate reports, using each
dataset's original gold and public Judge rubric. These are contributions of the
currently retrieved representations to Answer, including their context cost,
not an assessment of unreturned stored memories or a causal test of extraction
features such as prefetch. No contribution scores exist until the ablations run.

## Optional Answer/Eval

The optional evaluator consumes one completed retrieval artifact and never
repeats Add or Search. It reads the artifact's `unit` and dispatches to the
matching public AML pipeline for LongMemEval-S, LoCoMo-Refined, BEAM,
CLBench, or PersonaMem-v2:

```bash
export ANSWER_API_BASE='https://answer-endpoint/v1'
export ANSWER_API_KEY='...'
export ANSWER_MODEL='...'
export JUDGE_API_BASE='https://judge-endpoint/v1'
export JUDGE_API_KEY='...'
export JUDGE_MODEL='...'

python -m benchmark.aml.eval.evaluate \
  --retrieval result/aml-public/retrieval/longmemeval_s/retrieval.jsonl \
  --aml-repo result/aml-public-eval/agent-memory-leaderboard \
  --answer-concurrency 16 --eval-concurrency 32
```

This produces `answers.jsonl`, `judgments.jsonl`, and `summary.json` beside the
retrieval file. Answer prompts and scoring stay inside the pinned public
dataset pipeline; `evaluate.py` only selects it and adapts the generic
`retrieved_context` field where CLBench or PersonaMem expects a different input
shape. Answer and Judge each run up to 32 pipeline shards concurrently by
default; use `--answer-concurrency` and `--eval-concurrency` to tune them
independently. Outputs are merged back into retrieval order and published only
after every shard succeeds. These remain local proxy results rather than
official leaderboard scores. Model options and retries follow the selected
public pipeline; this wrapper does not provide the experiment-specific pacing
or resume logic. The pinned pipelines currently combine
an asynchronous HTTP client and a regular output file in one `async with`
statement; the wrapper corrects only that context-manager syntax in a temporary
runtime copy, leaving the verified checkout, prompts, and scoring code unchanged.

PersonaMem defaults to `--persona-mode mcq`, even for older retrieval files
marked as generative. Answer receives only retrieved memory, the question and
four shuffled choices from `correct_answer` plus `incorrect_answers`; preference
labels and the correct option marker are not added to the prompt. Search does
not receive these choices. The public pipeline maps the predicted letter to
its option text and grades it locally; invalid answers count as incorrect.
Only the three `ANSWER_*` variables are required for MCQ. Use
`--persona-mode generative` explicitly for the public narrow LLM judge instead.
Subprocesses default to `PYTHONHASHSEED=0` to keep option mappings stable across
shard counts. These are OV retrieval-based evaluations, not reproductions of
the authors' trained MemAgent implementation.

## Static checks

```bash
python -m ruff check benchmark/aml
python -m ruff format --check benchmark/aml
```
