# Trích xuất skeleton raw

Cập nhật: 01/11/2026. Phạm vi **video + metadata chính thức → raw NPZ + manifest**.
[Association](HAND_ASSOCIATION.md) · [QC](EXTRACTION_QC.md) · [Mac/front](EXTRACTION_MAC.md) ·
[tốc độ Pi 4](EXTRACTION_SPEED_PI4.md) · [Vision/cloud](EXTRACTION_VISION_CLOUD.md).

## Điểm và quy ước đã chốt

- **Representation mục tiêu 50 điểm:** 8 Pose + 21 mỗi tay. Pose theo thứ tự MediaPipe
  `[11,12,13,14,15,16,23,24]`: vai, khuỷu, cổ tay, hông trái/phải.
- Không lấy mặt, chân hoặc các điểm bàn tay thô của Pose; không chạy FaceMesh.
  Không tự tạo joint cổ; trung điểm hai vai có thể suy ra downstream.
- Pose + Hands Solutions riêng; config mặc định complexity=1, `smooth_landmarks=false`,
  `static_image_mode=false` (tracking detector vẫn hoạt động).
- Giữ XYZ image/world native; hệ Z/world của Pose và mỗi tay khác nhau. Không normalize,
  clip tọa độ, repair, resample hay augmentation; điểm thiếu giữ NaN.
- **Raw giữ tất cả hand candidates**, kể cả >2. Association L/R là view dẫn xuất riêng,
  không đổi raw; giữ nhãn/score detector và mask. Mask hữu hạn không chứng minh điểm đúng.
- `input_mirrored=false` ghi quy ước ảnh; không flip ảnh/tự đổi nhãn handedness.
- **Schema `vsl400.detector_raw.v3`**. Loader đọc v1 (33 Pose)/v2 (8 Pose), nhưng resume
  không dùng cache cũ thay v3. Thay topology/backend/config/policy cần identity phù hợp.

## Đầu ra NPZ

`T`: số frame; `D`: tổng hand candidates. XYZ/visibility/scores float32.

| Khóa | Shape / nội dung |
|---|---|
| `raw_pose_xyz`, `raw_pose_world_xyz`, `raw_pose_visibility` | `(T,8,3)`, `(T,8,3)`, `(T,8)` |
| `hand_candidate_xyz`, `hand_candidate_world_xyz` | `(D,21,3)`, toàn bộ candidates |
| `hand_candidate_label`, `hand_candidate_score` | `(D,)`, nhãn Unicode và score |
| `hand_candidate_offsets` | `(T+1,)` int64, ranh giới candidates theo frame |
| `associated_hand_xyz`, `associated_hand_world_xyz` | `(T,2,21,3)`, trái/phải giải phẫu |
| `association_indices/method/cost/margin/valid` | Các khóa riêng `(T,2)`; xem [association](HAND_ASSOCIATION.md) |
| `raw_hand_xyz`, `raw_hand_world_xyz` | `(T,2,21,3)`, detection slots khi N≤2; NaN khi N>2 |
| `handedness_label/score`, `pose_finite`, `hand_finite`, `detector_returned` | View/mask legacy; không dùng để đếm candidates v3 |
| `frame_indices` | `(T,)` int32 liên tiếp từ 0 |
| `timestamps`, `decoder_timestamps` | `(T,)` float64, giây |
| `meta_json` | JSON scalar: metadata gốc, identity, runtime/policy, completeness |

Timestamp ưu tiên decoder tăng nghiêm ngặt; fallback frame/FPS có warning (giả định CFR).
Không tự mặc định 25 FPS hoặc giả tạo world thiếu. Dùng `load_raw(path)` trong
[extract_raw.py](../extraction/extract_raw.py); downstream L/R đọc associated view kèm mask/method.

## Metadata, manifest và resume

- Tìm JSON `Part_N/front_view.json` cạnh `front_view/` (tương tự left/right).
  Giữ ID có số 0 đầu, signer/gloss; thiếu/sai metadata mặc định fail preflight.
- NPZ giữ `video_id`, Part/view/signer/gloss, `official_metadata`, hash JSON/video/code/config,
  mapping Pose, association policy/version và trạng thái hoàn tất. FPS thực tế và chính thức tách biệt.
- `manifest.csv` append theo sự kiện/lượt chạy, có `run_id`: `started` rồi `complete`,
  `incomplete`, `failed`, `skipped_complete`; lỗi preflight/runtime có trạng thái riêng.
  Tổng hợp sự kiện cuối mỗi clip; số dòng không phải số video.
- `batch_errors.jsonl` giữ lỗi/traceback và incomplete; lỗi một clip không dừng batch.
  NPZ ghi atomically; ledger flush xuống đĩa. Chỉ một writer cho mỗi output root.
- Decode complete đòi số frame khớp cả metadata chính thức và header nếu có;
  đạt `--max-frames` luôn chưa xác nhận complete. Ambiguity không làm decode incomplete.
- Resume chỉ skip complete đúng schema/nguồn/config/runtime/policy/metadata và số frame.
  Matching incomplete được thử lại; khác identity/corrupt cần output riêng hoặc `--overwrite`.
  Exit 1 khi có lỗi/incomplete. Không trộn cache giữa các điều kiện extraction.

## Chạy

Từ project root, dùng `.venv` riêng đã kiểm chứng; [setup/lock trên Mac](EXTRACTION_MAC.md):

```bash
.venv/bin/python extraction/extract_raw.py \
  dataset/raw/vsl400_input --output outputs/extraction/raw_v3 --view front --dry-run
```

- Pilot: thay `--dry-run` bằng `--max-videos 3`, chọn output riêng; full batch dùng `--resume`.
- `--view front|left|right` (mặc định all); `--max-videos N` chọn N video đầu sau lọc,
  gồm cả cache; `--max-frames N` giới hạn frame từng clip.
- `--association-policy PATH` đổi policy, mặc định [association_policy.json](../extraction/association_policy.json).
- Dry-run kiểm tra config/policy/metadata, không import MediaPipe/decode/tạo output.
  `--allow-missing-metadata` chỉ cho video ngoài dataset. macOS runtime cần truy cập đồ họa.

## Tiến trình và bằng chứng

- 30–31/10: raw v1; preflight 7 Part/3 view, 74.259 video khớp JSON.
- 01/11: v2 chọn Pose8 + tay42, thêm metadata/ledger/completeness/resume/dry-run.
  [Kiểm chứng v2](../outputs/extraction/verification_v2/summary.json); pilot Mac 9/14 complete,
  5 fail do guard >2 tay. Benchmark/QC xem tài liệu riêng.
- 01/11: v3 giữ mọi candidate + association có provenance; **14/14 complete, 883 frame**,
  9 frame có 3 candidates; replay offline khớp, 5 clip cần QC review. Chưa chạy full dataset.
  [Summary](../outputs/extraction/association_v3_verification/summary.json) ·
  [Tests](../outputs/extraction/association_v3_verification/tests.txt).
- 01/11: tạo venv riêng + dependency lock; 55 tests đạt, 70/70 clip complete và resume skip đúng.
  Preflight 24.753 front hợp lệ; [kiểm chứng môi trường](../outputs/extraction/venv_readiness_verification/summary.json).
  Có thể thu raw hàng loạt; association/QC vẫn cần review trước huấn luyện.

Thiết kế raw tách preprocessing tham khảo [SPOTER](https://arxiv.org/html/2210.00893v1),
[Roh et al.](https://aclanthology.org/2024.signlang-1.36.pdf); topology 50 điểm là lựa chọn dự án.
