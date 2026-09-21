# Associative evidence memory

Opt in with `memory.associative.enabled=true` and configure the existing OV VLM, embedding, and rerank providers. This pipeline and the older single-Event `memory.scene_cues` experiment are mutually exclusive. Defaults are unchanged.

## Build and retrieval

Each session commit independently segments its archived text into coherent scenes, assigns recurring specific topics, extracts atomic and connected facts, and generates four trigger families: Entity, Bridge, Scene, and Horizon. Scene and item evidence is written under each permitted user/peer `memories/associative/` root. Trigger text is only indexed, never returned as evidence. Existing OV profile extraction supplies optional standing context.

Topic updates use the upstream update prompt and preserve immutable revisions connected by a stable topic root. Later commits match against the latest revision, within a bounded prompt context. All versions remain searchable; they share one topic-selection budget. This avoids overwriting another commit's developments.

Models and storage use OV's configured providers. The prompt modules in `tmem_prompts/` are copied from Sherlockwz/T-Mem commit `dd9e1527bc75908485809580c9520af5a9a42879` under its MIT license (included there). The orchestration is implemented for OV's online commit, identity, task, and storage APIs rather than T-Mem's offline experiment directories.

Indexing retains upstream field weighting: topic title ×2; scene subject ×3 / summary ×2; fact content ×3 / generated query patterns ×2 for BM25. Dense fields and per-kind rerank text follow the upstream composition. Generated query patterns and triggers stay out of answer evidence.

Retrieval follows topic -> scene -> item links; each layer combines BM25 and dense ranks, then reranks grounded evidence. A separate raw-dialogue/Scene/Horizon three-way RRF channel expands scene candidates. Entity/Bridge use concept, bridge, and joint views with confidence filtering and a cosine gate, expanding fact candidates. Defaults match upstream candidate sizes (250 / 15 topics / 24 scenes / 40 facts; final 5 scenes / 15 facts), with one optional OV profile. The requested limit remains an upper bound.

Both `find` and `search` use this pipeline for explicit memory scopes when enabled. Resource/skill/image and mixed untyped scopes keep their existing routing. The external AML `/search` need not change its underlying `find` call. A configured reranker is required by default, even for the otherwise QUICK find route. With `rerank_required=true`, provider errors or invalid scores fail the read request so callers can retry; they do not silently change evaluation into vector-only retrieval. Existing namespaces without an associative corpus fall back to normal OV retrieval.

## Reliability and isolation

Accepted model responses, source snapshots, embeddings, and the materialization manifest are persisted before commit completion. A retry reuses them instead of resampling or duplicating evidence. The native commit task tracks the independent associative step. No existing archives are automatically backfilled.

Self-only archives retain leading and assistant-only batches. In peer/mixed histories, only permitted self/peer user turns and their attributable assistant continuations enter each target's build; ambiguous leading replies are not broadcast across identities. The index shares OV account/namespace/ACL filters; evidence is checked against current file and source hashes and filesystem permissions before reranking/returning it. Linked facts also validate their source-scene hashes. Topic and trigger metadata do not enter answer context.

Delete/account cleanup and move/copy maintain auxiliary index records. Source provenance remains attached on copy; inaccessible or changed sources fail closed. Oversized single messages fail explicitly without truncation; scene and overlapping fact-extraction batches bound model inputs; commit boundaries are flush boundaries. Item extraction can combine prior scenes of the same topic, within its bounded context.

This is a migration of the memory/retrieval mechanisms into OV, not a claim of identical T-Mem benchmark scores. Provider models, a dependency-free Latin/CJK tokenizer and positive-IDF BM25 implementation (instead of NLTK/Okapi), commit boundaries, immutable topic versions with stable topic roots, OV profiles, native API response formatting, and the benchmark's Answer/Judge remain OV integrations. Compare complete-pipeline evaluation before attributing individual gains.
