# GRASP Gold Evidence Builder

Independent benchmark-annotation to passage-ID mapping for one pinned revision of
`doancuavahk/grasp-retrieval`. There is no import dependency on the GRPO trainer.
See [RUN_VAST_BUILDER.md](RUN_VAST_BUILDER.md) to run a fresh clone on Vast/Linux,
and [HUONG_DAN_LOCAL.md](HUONG_DAN_LOCAL.md) for existing local artifacts.

## Sources and mapping

- HotpotQA: labeled train/dev from `hotpotqa/hotpot_qa`, pinned HF revision; original supporting facts.
- 2Wiki: official train/dev archive; original supporting facts. The SPARKLE OSF example was checked
  first but exposes evaluation QA rather than the required training QA.
- NQ: official DPR gold-info train/dev annotations with original gold context, short answers,
  source example ID and Wikipedia URL/title. Raw full NQ HTML is not downloaded.

HotpotQA/2Wiki supporting titles must map uniquely in their exact passage DB. Title matching
preserves punctuation so `C`, `C++` and `C#` remain distinct. Processed `gold_ctxs`
are also accepted when IDs are compatible. NQ raw gold contexts require the dedicated alignment
step: exact normalized token containment, >=20 tokens, URL/title agreement and auditable offsets.
Ambiguous duplicate matches, near matches and unmapped evidence are never labeled as mapped. NQ's corpus oldid or
original HTML offsets cannot be recovered from the current passage DB and are never fabricated.

Gold labels are never inferred from retrieval output, answer-string searches, or model-generated
triples. Original source/derived data hashes and excluded questions are retained in reports.

## Installation

On Linux (Python 3.11+), install independently of the trainer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[download]'
```

On Windows, create a Python environment inside this repo (Python 3.11+):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[download,dev]"
```

## Exact-QA workflow

Download public source annotations once (does not download the retrieval corpus or private QA):

```bash
grasp-gold-fetch --dataset hotpotqa --split all --raw-root raw
grasp-gold-fetch --dataset 2wiki --split all --raw-root raw
grasp-gold-fetch --dataset nq --split all --raw-root raw
```

Build on Linux with the original QA and the exact pinned retrieval snapshot supplied separately:

```bash
grasp-gold-pipeline --qa /workspace/data/grpo_train_mixed_shuffled.jsonl \
  --retrieval-root /workspace/data/grasp-retrieval \
  --retrieval-revision 28658c8f6442c3aebb5abc6974a52184dca59e9e \
  --expected-train-rows 9000
grasp-gold-package
```

Windows wrapper for the same pipeline:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_for_grpo.ps1 `
  -QaFile "C:\data\grpo_train_mixed_shuffled.jsonl" `
  -RetrievalRoot "C:\data\grasp-retrieval" `
  -RetrievalRevision "28658c8f6442c3aebb5abc6974a52184dca59e9e"
```

The wrapper prefers this repo's `.venv`, or accepts `-PythonExecutable`. No other code repo is
required. Original QA, six raw annotation files and three `passages.sqlite` files are required;
Git clone does not provide these ignored data artifacts. Use `--help` on any command for paths.

The file contains 9,000 shuffled questions. `--qa` joins question AND answer, even if a source ID
matches; synthetic IDs are not assumed to be benchmark IDs. Official source IDs are preserved in
canonical output, paired with input IDs. `--retain-unmapped --qa <QA-file>` preserves every input
row in order, with `input_id` and `evidence_status`. Unmapped rows have `gold_doc_ids: []` and
`evidence_reason`; they are retained as QA without fabricating evidence or changing rewards.
The Windows wrapper enables this flag for the 9,000-question train input.

`scripts/fetch_official_raw.py --dataset {hotpotqa,2wiki,nq} --split all --raw-root raw` downloads
annotations and registers revisions, schema, source split and SHA256. HotpotQA defaults to HF
because the old CMU download hostname may fail DNS. The original CMU URL remains an explicit option.

`grasp-gold-align-nq --raw-root raw --retrieval-root <retrieval> --qa <existing-QA> --output-dir work/nq`
creates processed NQ files and full alignment/exclusion proofs. `grasp-gold-build --source
wiki18=work/nq/train.json` then consumes those IDs. Running build directly on unaligned NQ gold-info
does not admit its rows. Do not treat arbitrary DPR positive/retrieved contexts as original gold.

## Artifacts

Canonical rows have `id`, `dataset`, `split`, `question`, `gold_answers`, `gold_doc_ids`,
`evidence_status` (`mapped` or `unmapped`). Only mapped rows may contain gold IDs, and mapped
rows must contain at least one verified ID. Unmapped IDs use a dedicated namespace; `input_id`
retains the original input identity, and `source_id` is included when the annotation was found.
Dataset keys: `2wiki_ircot`, `hotpotqa_abstracts_2017`, `wiki18`. Train/dev files and mapping reports
are staged under `work/exact-grpo`; the manifest pins SQLite and QA hashes. Validation rejects empty
files, inconsistent status/labels, malformed splits/IDs, nonexistent passage IDs and cross-split
question overlap. `train.jsonl` retains all 9,000 questions (6,374 mapped, 2,626 unmapped in
the current rebuilt artifact; see the manifest for exact counts and hashes).
Validate with `--mapped-train-output work/exact-grpo/train_mapped.jsonl` to emit the exact
mapped subset and record both file hashes and coverage in the manifest. Recall-aware training
uses this subset; recall is undefined for unmapped rows. Dev remains a mapped-only subset.

`grasp-gold-package` (or `python scripts/package_grpo_handoff.py`) produces the standalone data-only
`work/handoff/grasp-gold-evidence-v1.zip`. It verifies QA/validation hashes, exact mapped subset and
NQ report/corpus/source consistency. The alignment report is found inside the selected output
directory; it does not depend on a previous local run's `work/nq-exact` directory.

Only when a combined code/data ZIP is wanted, explicitly provide the trainer clone:

```bash
grasp-gold-package --trainer-dir ../grasp-grpo-trainer --output work/handoff/grasp-evidence-grpo-v2.zip
```

Run packaging before the real SFT smoke gate (30 questions/corpus); it stages report hashes into
the manifest. Preserve the successful gate before publication/primary GRPO. Unchanged repackaging
preserves the manifest hash. Git ignores raw sources, output QA, archives, indexes, credentials and
Python environments; transfer data separately. To verify code: `python -m pytest` and `ruff check .`.
