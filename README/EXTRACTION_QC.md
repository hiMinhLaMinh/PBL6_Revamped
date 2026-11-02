# Quality control cho raw extraction

Cập nhật: 01/11/2026. QC hậu kiểm; không sửa/xóa raw hoặc tự loại clip.

## Ba lớp kiểm tra

1. **Integrity:** schema/mask/dtype, frame count chính thức + header, completeness,
   source identity, lỗi/chạy dở trong manifest và coverage dataset.
2. **Quan sát/heuristics:** NaN theo joint, visibility Pose, số hand candidates,
   đoạn không thấy tay, XY ngoài ảnh, timestamp fallback và ambiguity/uncertainty khi association.
3. **Visual QA:** video + overlay; preview đầu/giữa/cuối không thay cho xem toàn chuyển động,
   đặc biệt crossing/occlusion/ngón tay nhanh. Frame overflow cần soi riêng.

[quality_control.py](../extraction/quality_control.py) tạo `qc_report.csv`,
`qc_details.jsonl`, `qc_summary.json`, `batch_failures.csv` và preview tùy chọn.
Batch failures lấy từ sự kiện cuối mỗi clip, kể cả chưa có NPZ. `--dataset-root` so với
mọi front_view.json; pilot không đủ coverage phải có `dataset_complete=false`.
Một output root chỉ chứa một extraction condition.

## Policy và diễn giải

[qc_policy.json](../extraction/qc_policy.json), hash lưu trong báo cáo: visibility cutoff 0.5;
review nếu Pose quan sát <0.9, không thấy tay >0.2 hoặc XY ngoài ảnh >0.05.
Đây là **ngưỡng khởi đầu chưa hiệu chuẩn**. Association có [policy riêng](HAND_ASSOCIATION.md).

- V3 đếm tay/XY trên **mọi candidate** qua offsets, không dùng legacy slots khi overflow.
- Bổ sung flags `hand_candidate_overflow`, `association_ambiguous`,
  `association_tracking_uncertain`; báo missingness view L/R riêng với raw slots.
- `not_flagged` không có nghĩa tọa độ đúng; không ép mọi ký hiệu có hai tay.
  Handedness score không phải confidence hình học từng joint.
- Chưa dùng velocity theo detector slot vì thứ tự có thể đổi. World/Z có hệ riêng;
  biên ảnh chỉ xét XY hữu hạn. Gán bằng Pose cũng chưa phải ground truth handedness.
- Preview v3: candidates xám, tay trái giải phẫu xanh lá, tay phải xanh dương;
  xác minh hash video nguồn, lưu seed/ánh xạ NPZ → ảnh.
  Sampling kết hợp clip flagged và ngẫu nhiên theo Part/signer.

## Chạy

Từ project root, dùng Python môi trường extraction:

```bash
python extraction/quality_control.py outputs/extraction/front_v3 \
  --output outputs/extraction/front_v3_qc --dataset-root dataset/raw/vsl400_input \
  --previews 24 --seed 42
```

Policy QC đổi được mà không gọi MediaPipe. Exit 1 nếu NPZ invalid, batch chưa complete
hoặc preview lỗi; review flags không tự loại clip. Coverage thiếu được ghi riêng;
kiểm tra dataset_complete trước khi khóa dataset.

## Tiến trình và việc còn lại

Pilot v2: 9/14 complete, 5 fail do giới hạn >2 detections. V3 đã giữ mọi candidates:
**14/14 complete (883 frame), 9 frame overflow, 5 clip review**, không còn batch lỗi.
Replay association khớp cả 14 NPZ; QC không đổi hash raw. Đã xem 2 frame overflow riêng,
candidates gần/chồng nhau; chưa gán nhãn accuracy hoặc duyệt hết chuyển động.

Pilot mở rộng trong venv riêng: 70/70 complete, 43 not_flagged, 27 review, không có batch lỗi;
7 signer từ 10 clip đầu mỗi Part, chưa đại diện mọi signer/gloss. Đây là kiểm chứng vận hành,
không phải accuracy audit. [Bằng chứng](../outputs/extraction/venv_readiness_verification/summary.json).

Trước huấn luyện: visual QA có hướng dẫn/nhãn review, hiệu chuẩn và freeze policy bằng
train/validation, ghi quyết định loại mẫu, khóa split đúng protocol, tránh chọn bằng test.
Cần thêm duplicate-video audit và đánh giá accuracy landmark khi có ground truth.

[QC v3](../outputs/extraction/association_v3_verification/qc/qc_summary.json) ·
[Frame overflow](../outputs/extraction/association_v3_verification/overflow_review/index.json) ·
[Tests](../outputs/extraction/association_v3_verification/tests.txt).
