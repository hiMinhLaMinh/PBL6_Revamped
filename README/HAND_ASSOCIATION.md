# Hand association và handedness

Cập nhật: 01/11/2026. **Đã triển khai trong schema raw v3**, tách dữ liệu detector
khỏi phép gán bên tay để thay policy/ablation mà không chạy lại MediaPipe.

## Quyết định kỹ thuật

- Giữ **mọi hand candidate**: XYZ image/world, handedness label và score nguyên gốc.
  Lưu mảng phẳng cùng `hand_candidate_offsets (T+1,)`; không pickle/object array.
- Representation mục tiêu là 8 Pose + 42 tay. Detector có thể trả 0/1/2/3+ candidates;
  không cắt hai candidate đầu, không làm clip fail chỉ vì thừa detection.
- [hand_association.py](../extraction/hand_association.py) xét toàn bộ N candidates,
  gán một-một cho **trái/phải giải phẫu**, có lựa chọn không gán.
- Ưu tiên cổ tay Pose có visibility đủ ngưỡng; fallback vị trí đã gán trước đó,
  tối đa 5 frame kể từ lần xác nhận bằng Pose. Tracking không tự gia hạn xác nhận.
- Không dùng nhãn handedness làm anchor mặc định; label/score vẫn giữ để nghiên cứu sau.
  Không ép đủ hai tay, dedup, nội suy hoặc đổi tọa độ raw.
- Candidate xa anchor hoặc các phương án gần tương đương: abstain, tọa độ NaN và cờ
  `ambiguous`. Hai tay chạm nhau không đủ căn cứ tự kết luận duplicate.

Tái sử dụng nguyên tắc Pose → tracking của PBL6 cũ; thay helper chỉ xét candidates 0/1
bằng matching N×2. Các ngưỡng hiện là **heuristic chưa hiệu chuẩn**, không phải ground truth.

## Dữ liệu và provenance

| Khóa | Ý nghĩa |
|---|---|
| `hand_candidate_xyz/world_xyz`, `hand_candidate_label/score` | Toàn bộ raw candidates |
| `hand_candidate_offsets` | Candidates frame i nằm trong `[offsets[i]:offsets[i+1]]` |
| `associated_hand_xyz/world_xyz` | `(T,2,21,3)`, bên trái rồi phải giải phẫu |
| `association_indices` | Index candidate trong từng frame; -1 là không gán |
| `association_proposed_indices` | Phương án tốt nhất trước khi từ chối vì ambiguity |
| `association_method` | `none`, `pose`, `tracking`, `ambiguous` |
| `association_cost`, `association_margin` | Khoảng cách của phép gán được nhận; chênh cost với phương án khác |
| `association_valid` | Chỉ true khi gán được bằng Pose; không phải xác suất chính xác |
| `associated_hand_observed` | Mask điểm hữu hạn, không phải nhãn đúng/sai |

Các tên `xyz/world_xyz`, `label/score` trong bảng là viết gọn cho hai khóa riêng.
Legacy `raw_hand_xyz` giữ detection slots khi N≤2; N>2 để NaN, đọc candidates thay vì coi
đó là mất tay. Downstream cần L/R dùng `associated_hand_xyz`, vẫn kiểm tra method/mask.

[association_policy.json](../extraction/association_policy.json): visibility≥0.5,
Pose gate 0.15, tracking gate 0.10, ambiguity margin 0.015, TTL 5 frame.
Khoảng cách theo đơn vị chiều rộng ảnh, Y nhân height/width. Policy/version và hash code
được lưu cùng NPZ; đổi policy/schema/code không resume nhầm cache cũ.

## Chạy lại riêng association

Từ project root, dùng Python môi trường extraction:

```bash
python extraction/hand_association.py path/to/clip.mp4.raw.npz \
  --policy extraction/association_policy.json --output outputs/extraction/reassociated/clip.npz
```

Output là **NPZ dẫn xuất mới**, chứa phép gán, frame indices và hash NPZ nguồn;
raw không đổi. Cache v1/v2 vẫn đọc được nhưng thiếu toàn bộ candidates, cần trích xuất lại
để có v3. Decode completeness độc lập với chất lượng association.

## Kiểm chứng và tiến trình

01/11: bỏ guard >2, thêm lưu mọi candidate, association, replay offline và QC.
55 tests đạt: 0/1/2/3+ tay, candidate đúng ở vị trí thứ ba, đảo thứ tự, ambiguity,
tracking hết hạn, một Pose wrist, aspect ratio, schema và bất biến raw.
Pilot **14/14 clip complete, 883 frame**, gồm 5 clip từng fail; giữ đủ 9 frame có 3 candidates.
Replay khớp mọi mảng association của cả 14 NPZ; QC không đổi hash raw, 5 clip cần review.
Đã xem trực tiếp 2 frame overflow có candidates chồng gần nhau; chưa đánh giá accuracy
landmark hoặc duyệt toàn chuyển động, chưa chạy toàn dataset.

[Bằng chứng](../outputs/extraction/association_v3_verification/summary.json) ·
[QC](EXTRACTION_QC.md) · [Extraction](EXTRACTION.md).
