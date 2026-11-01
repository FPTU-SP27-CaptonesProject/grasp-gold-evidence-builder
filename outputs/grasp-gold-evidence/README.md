---
language: en
task_categories:
  - question-answering
license: other
---
# GRASP gold evidence (local release candidate)

Retrieval revision: `28658c8f6442c3aebb5abc6974a52184dca59e9e`.
Full train: 9000; mapped: 6374; unmapped: 2626.
Dev: 21249. Counts describe this artifact, not full benchmark coverage.

train.jsonl retains the input QA in order with input_id. evidence_status is mapped or unmapped.
Mapped rows have verified gold_doc_ids. Unmapped rows have [] and evidence_reason; recall is
unavailable, not zero. train_mapped.jsonl is the exact mapped-only subset for recall-aware GRPO.
Do not infer gold labels from retrieved rankings, answer-string search or generated triples.
Gold IDs are used only after rollout for reward/evaluation, never in model prompts.

HotpotQA/2Wiki use original supporting-fact titles with punctuation-preserving title matching.
NQ uses original DPR gold contexts and Wikipedia URL/title provenance. Exact normalized-token
containment (minimum 20 tokens) verifies passage content. Proofs preserve token offsets and hashes.
Original HTML offsets and the corpus Wikipedia oldid are unavailable and are never fabricated.
Only public labeled dev annotations are supplied; no hidden-test labels are fabricated.

Source URLs, revisions, SHA256 and attribution are in manifest.json/source_inventory.json.
HotpotQA: https://github.com/hotpotqa/hotpot (CC BY-SA 4.0).
2Wiki: https://github.com/Alab-NII/2wikimultihop/blob/main/LICENSE (source terms).
NQ: https://github.com/google-research-datasets/natural-questions (annotation terms).
DPR: https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py (source/licensing).
Wikipedia-derived material retains the original attribution/share-alike requirements.
This package supplies QA labels and reports; it does not duplicate the retrieval corpus.

Release candidate: canonical data validation passed; real SFT rollout smoke is pending.
Run smoke (30 questions/corpus, nonzero context recall and reward variance) before primary GRPO.
Package before smoke; changing the manifest afterwards invalidates its hash binding.
Local preflight and data validation are separate checks from the real model rollout gate.
