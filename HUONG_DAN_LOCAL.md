# Dữ liệu evidence đã chuẩn bị từ file QA 9.000 câu

File đầu vào chính xác là:
`../grasp-grpo-trainer/grpo_train_mixed_shuffled.jsonl`.
Đây chính là file QA, không cần tìm thêm file mang tên QA khác.

Code dùng annotations gốc để ghép đúng câu hỏi và đáp án. `train.jsonl` giữ đủ 9.000 câu theo thứ tự
đầu vào, thêm `input_id` để đối chiếu. Mỗi câu có `evidence_status=mapped` hoặc `unmapped`.
Câu unmapped có `gold_doc_ids=[]` và `evidence_reason`; không tạo gold labels từ retrieval/triples.

## Artifacts đã tạo

```text
work/exact-grpo/
  train.jsonl                       # Đủ 9.000 câu: 6.374 mapped + 2.626 unmapped
  train_mapped.jsonl                # 6.374 câu mapped, dùng cho recall-aware GRPO
  dev.jsonl                         # 21.249 câu
  manifest.json
  source_inventory.json
  reports/
    train_mapping_report.json
    dev_mapping_report.json
    validation_report.json
    nq_alignment_report.json        # được đưa vào gói bàn giao
  nq-aligned/
    train.json
    dev.json
    alignment_report.json
```

Train giữ 3.000 câu mỗi corpus. Trong đó mapped gồm 2.996 câu 2Wiki, 2.993 câu HotpotQA,
385 câu NQ. Unmapped tương ứng 4, 7 và 2.615 câu. `train_mapped.jsonl` là phép lọc chính xác
`evidence_status == "mapped"` từ train đầy đủ, không đổi thứ tự. Dev hiện chỉ có các câu mapped;
NQ dev có 1.313 câu và phải ghi rõ phạm vi tập con khi báo cáo nghiên cứu.

Các số liệu trên được tạo lại sau khi sửa chuẩn hóa title để giữ dấu câu. Những title khác nhau
không còn bị gộp do xóa dấu câu. SHA256 mới trong manifest xác định đúng phiên bản dữ liệu;
không dùng lại smoke report của phiên bản trước.

Không chấm recall=0 cho unmapped: thiếu nhãn không chứng minh retrieval sai. Cấu hình trainer
dùng `train_mapped.jsonl` cho reward `0.3*F1 + 0.7*recall`. Giữ đủ 9.000 QA trong output không
có nghĩa cả 9.000 đều đủ nhãn để chạy reward này. Muốn huấn luyện cả các câu unmapped cần một
chế độ reward riêng được thiết kế và báo cáo rõ; pipeline hiện không tự đổi sang F1-only.

## Mapping NQ

Raw NQ gold contexts/provenance đã tải từ DPR (train/dev). Adapter quét Wiki18, kiểm tra URL/title
và tìm đoạn token liên tiếp khớp đầy đủ: gold span nằm trong passage hoặc passage nằm trong gold
span, tối thiểu 20 token. Không dùng overlap gần đúng để gán evidence. Report ghi token offsets,
text hash và URL chứa oldid. Corpus không có HTML offsets/oldid nên adapter không tuyên bố đã
khôi phục những trường này; nó xác minh nội dung passage trong snapshot local đã pin hash.

## Chạy lại nếu thay input

```powershell
Set-Location D:\FPTU\FPTU-SP26-CaptonesProject\grasp-gold-evidence-builder
powershell -ExecutionPolicy Bypass -File .\scripts\build_for_grpo.ps1 `
  -QaFile "..\grasp-grpo-trainer\grpo_train_mixed_shuffled.jsonl"
```

Script dùng raw annotations đã tải. Chỉ thêm `-Download` khi cần lấy nguồn trên một máy mới.
Tạo `.venv` và cài repo theo README trước. Có thể truyền `-PythonExecutable` và `-RetrievalRoot`
để chọn Python/corpus; không còn phụ thuộc môi trường Python của repo proxy.
Nó dựng lại NQ alignment rồi canonical train/dev và manifest. Xem mapping reports, QA input hash
và `validation_report.json`; yêu cầu `valid=true`, `invalid_doc_ids=0`.

## Đưa lên Vast

Output được phép push nằm ở `outputs/grasp-gold-evidence/`, gồm ba JSONL, manifest, README,
source inventory và bốn báo cáo. `work/` vẫn là thư mục làm việc và được bỏ qua.
Sau mỗi lần build, chạy `grasp-gold-package --git-output-dir outputs/grasp-gold-evidence` để
cập nhật snapshot rồi commit. Clone repo trên Vast sẽ có snapshot này, không cần upload ZIP:

```bash
mkdir -p /workspace/data
cp -a outputs/grasp-gold-evidence /workspace/data/
```

Nếu dùng cách chuyển ZIP: push repo `grasp-grpo-trainer`, clone trên Vast rồi upload gói dữ liệu
`work/handoff/grasp-gold-evidence-v1.zip`. Giải nén ZIP này vào `/workspace` để tạo
`/workspace/data/grasp-gold-evidence/`. Gói chứa canonical QA và reports; không chứa code/model/index.
Đọc `../grasp-grpo-trainer/RUN_VAST_EVIDENCE_V2.md` để clone vào thư mục mới, cấu hình reader,
SFT checkpoint và retrieval snapshot đang có, rồi chạy preflight, SFT smoke và pilot GRPO.
Không cần train SFT lại. ZIP kết hợp `work/handoff/grasp-evidence-grpo-v2.zip` vẫn có nếu cần
chuyển cả code/data mà không clone, nhưng không cần upload cả hai ZIP.

Tạo ZIP dữ liệu độc lập bằng `grasp-gold-package` sau khi build/validate; không cần có repo trainer
nằm cạnh. Nếu muốn tạo ZIP kết hợp, dùng `grasp-gold-package --trainer-dir ../grasp-grpo-trainer`.
Để clone builder và chạy lại toàn bộ trên Linux, xem `RUN_VAST_BUILDER.md`.

Data validation không thay thế real rollout smoke. Chưa publish dataset hoặc chạy primary GRPO
trước khi smoke 30 câu/corpus đạt recall và reward variance. Dùng dev để chọn checkpoint;
không tạo test.jsonl từ test không có nhãn.
