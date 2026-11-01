# Chạy evidence builder độc lập trên Vast/Linux

Repo này chỉ tạo QA/evidence, không train model và không cần GPU. Không cần clone trainer để build
hoặc đóng ZIP dữ liệu. Chạy các lệnh dưới đây trong Terminal, từ thư mục repo đã clone.

## 1. Cài Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[download]'
```

Yêu cầu Python >=3.11. Repo có code và snapshot QA/báo cáo trong `outputs/grasp-gold-evidence/`
khi đã commit output. `raw/`, `work/`, corpus, checkpoint và ZIP vẫn không được push.

Nếu dùng luôn snapshot đã commit, không cần build lại: copy `outputs/grasp-gold-evidence/`
vào `/workspace/data/` trên máy trainer. Các bước tiếp theo dành cho việc tạo lại evidence.

## 2. Đưa input lên máy

Chuẩn bị file QA gốc và retrieval snapshot:

```text
/workspace/data/grpo_train_mixed_shuffled.jsonl
/workspace/data/grasp-retrieval/
  2wiki_ircot/passages.sqlite
  hotpotqa_abstracts_2017/passages.sqlite
  wiki18/passages.sqlite
```

Retrieval phải đúng commit `28658c8f6442c3aebb5abc6974a52184dca59e9e` của
`doancuavahk/grasp-retrieval`. Dùng tài khoản HF có quyền đọc dataset private để tải snapshot này.
Builder chỉ cần SQLite; trainer dùng thêm FAISS index và triple cache riêng.
Giữ file QA gốc đủ 9.000 câu, không thay bằng file mapped-only đã tạo.

## 3. Tải annotations gốc

```bash
grasp-gold-fetch --dataset hotpotqa --split all --raw-root raw
grasp-gold-fetch --dataset 2wiki --split all --raw-root raw
grasp-gold-fetch --dataset nq --split all --raw-root raw
```

HotpotQA dùng một HF commit cố định; revision, URL và SHA256 được lưu trong
`raw/source_metadata.json`. NQ tải DPR gold-info train/dev, không tải toàn bộ NQ HTML.
Nếu đã chuyển đầy đủ `raw/` và metadata từ local, không cần tải lại.

## 4. Build và validate

```bash
grasp-gold-pipeline \
  --qa /workspace/data/grpo_train_mixed_shuffled.jsonl \
  --retrieval-root /workspace/data/grasp-retrieval \
  --retrieval-revision 28658c8f6442c3aebb5abc6974a52184dca59e9e \
  --expected-train-rows 9000 \
  --output-dir work/exact-grpo
```

Pipeline chạy inventory → NQ alignment → train/dev mapping → validation và mapped subset.
`train.jsonl` giữ đủ 9.000 câu theo thứ tự input, có `evidence_status=mapped/unmapped`.
Unmapped có `gold_doc_ids=[]` và lý do; không tự gán evidence. `train_mapped.jsonl` chỉ chứa mapped.
Validation phải có `valid=true`, `invalid_doc_ids=0`, `invalid_rows=0`.

## 5. Đóng ZIP và bàn giao

```bash
grasp-gold-package --evidence-dir work/exact-grpo --git-output-dir outputs/grasp-gold-evidence
```

Tải về `work/handoff/grasp-gold-evidence-v1.zip`. Gói chứa QA, manifest, README và các báo cáo;
không chứa raw annotations, model, SQLite, FAISS hay code trainer.
Giải nén vào `/workspace` trên máy trainer để có `/workspace/data/grasp-gold-evidence/`.

Lệnh trên cũng tạo snapshot `outputs/grasp-gold-evidence/` để commit lên GitHub. Sau khi push,
clone repo sẽ có QA/báo cáo ngay; có thể dùng snapshot này thay cho upload ZIP:

```bash
mkdir -p /workspace/data
cp -a outputs/grasp-gold-evidence /workspace/data/
```

Trainer dùng `train_mapped.jsonl` cho reward có recall; `train.jsonl` đầy đủ dùng để bàn giao và
tiếp tục bổ sung mapping. Thiếu nhãn evidence nghĩa là chưa chấm được recall, không phải recall=0.
Chạy preflight và SFT rollout smoke trên trainer trước primary GRPO. Builder chưa tạo checkpoint
model hoặc kết quả smoke GPU.
