# Trích xuất skeleton raw

Cập nhật: 31/10/2026. Phạm vi: **video → cache landmark gốc** để dùng lại cho ablation.
Thảo luận riêng: [tốc độ Pi 4](EXTRACTION_SPEED_PI4.md), [Vision API/cloud](EXTRACTION_VISION_CLOUD.md).

## Quyết định kỹ thuật

| Lựa chọn hiện tại | Lý do / quy ước |
|---|---|
| **75 điểm: 33 Pose + 21 mỗi tay**, bỏ FaceMesh | Giữ đầy đủ đầu ra Pose/Hands; chọn tập điểm cho model ở bước sau. Pose vẫn có các điểm mặt thuộc topology của nó. |
| MediaPipe Solutions **Pose + Hands riêng**, phiên bản 0.10.14 | Giữ backend rõ ràng; đây không phải Holistic. Config hiện tại dùng complexity=1 cho cả hai. |
| Lưu XYZ native và world estimates | Cho phép khảo sát XY/XYZ sau này. XY chuẩn hóa theo kích thước ảnh; Z và world của Pose/mỗi tay có hệ quy chiếu riêng, không phải hệ 3D chung đã hiệu chuẩn. |
| Không normalize theo cơ thể/tay, repair, resample hay augment | Cache giữ quan sát trước feature engineering. Không clip tọa độ về `[0,1]`. |
| `smooth_landmarks=false`, `static_image_mode=false` | Tắt smoothing Pose nhưng vẫn giữ tracking nội bộ detector. Thay cấu hình detector cần trích xuất thành cache khác. |
| Thiếu tọa độ → **NaN**; giữ visibility nguyên bản | Chưa áp threshold, điền zero hay copy frame. Mask hữu hạn chỉ phản ánh có tọa độ, không chứng minh điểm đúng. |
| Hai tay lưu theo **detector slots** | Thứ tự có thể đổi giữa frame; chưa gán trái/phải giải phẫu. Giữ label/score handedness để xử lý sau. |
| Giữ mọi frame đọc được và timestamp gốc | Ưu tiên timestamp decoder tăng nghiêm ngặt; nếu không hợp lệ, dùng frame/FPS và ghi warning (giả định CFR). Không tự mặc định 25 FPS. |
| Schema riêng: `vsl400.detector_raw.v1` | Raw cũ chỉ có 49 điểm XY. Không dùng lẫn cache mới với schema đặc trưng đã normalize của pipeline cũ. |

`input_mirrored=false` chỉ ghi quy ước input, không flip ảnh hay tự đổi handedness.
Score handedness là confidence phân loại cả tay, không phải confidence từng joint.
Trước khi dùng chuỗi tay cho model, cần gán và theo dõi bên tay rõ ràng.

## Đầu ra NPZ

`T` là số frame lưu; 75 là số vị trí/frame, không bảo đảm mọi điểm đều được phát hiện.
Pose dùng index `0..32`, mỗi tay `0..20` theo MediaPipe; giữ riêng cổ tay từ hai detector.

| Khóa | Shape / kiểu | Nội dung |
|---|---|---|
| `raw_pose_xyz`, `raw_pose_world_xyz` | `(T,33,3)` float32 | Pose image/world estimates |
| `raw_hand_xyz`, `raw_hand_world_xyz` | `(T,2,21,3)` float32 | Hai hand slots; world thiếu cũng là NaN |
| `raw_pose_visibility` | `(T,33)` float32 | Visibility chưa threshold |
| `handedness_label`, `handedness_score` | `(T,2)` Unicode / float32 | `Left`/`Right`/rỗng và score/NaN |
| `pose_finite`, `hand_finite` | `(T,33)` / `(T,2,21)` bool | Tọa độ XYZ hữu hạn |
| `detector_returned` | `(T,3)` bool | `[pose, hand_slot_0, hand_slot_1]` |
| `frame_indices` | `(T,)` int32 | Liên tiếp từ 0 |
| `timestamps`, `decoder_timestamps` | `(T,)` float64 | Giây; timeline được chọn và decoder giữ riêng |
| `meta_json` | scalar Unicode | Config, nguồn video, runtime, timing và warnings |

Metadata ghi SHA-256 video/code, đường dẫn nguồn, kích thước ảnh, FPS và giới hạn pilot.
`--resume` chỉ skip khi schema và identity video/config/runtime/`max_frames` khớp.
Ghi file atomically, không dùng pickle; giữ cấu trúc thư mục và đuôi video để tránh trùng Part/view.

## Chạy và đọc dữ liệu

File chính: [extract_raw.py](../extraction/extract_raw.py), [raw_extraction.json](../extraction/raw_extraction.json),
[requirements.txt](../extraction/requirements.txt). Python đã kiểm chứng: 3.11; các thư viện chính đã pin.

```bash
cd /Users/vohoangminh/PBL6/PBL6_Revamped/extraction
/Users/vohoangminh/PBL6/PBL6/.venv/bin/python extract_raw.py \
  /Users/vohoangminh/PBL6/PBL6/014170.mp4 --output ../outputs/extraction/raw_cache --resume
```

Truyền thư mục thay video để chạy batch đệ quy. Pilot dùng `--max-frames 12` và output riêng.
`--overwrite` cho phép thay cache; batch có video lỗi sẽ tiếp tục và kết thúc với exit code 1.
Giữ nguyên input root khi resume. Trên macOS, lỗi `NSOpenGLPixelFormat` có thể cần chạy
trong phiên desktop thông thường.

```python
from extract_raw import load_raw
arrays, meta = load_raw("../outputs/extraction/raw_cache/014170.mp4.raw.npz")
pose_xy = arrays["raw_pose_xyz"][..., :2]
hand_xy = arrays["raw_hand_xyz"][..., :2]  # Chưa gán bên tay.
```

## Tiến trình và kiểm chứng

- **30/10:** kiểm tra raw cũ, chốt 75 điểm, triển khai extractor/config/validator/tests;
  chuyển phần việc sang `PBL6_Revamped`.
- **Kiểm chứng ngày 30/10:** 13 tests đạt; clip `014170.mp4` đọc đủ **54/54 frame**, 25 FPS,
  1080×1080; load/schema/masks, code hash và resume hợp lệ. Pose/hand slot 0/slot 1
  được trả về lần lượt ở 54/54/45 frame; đây không phải thống kê độ chính xác.
- Bằng chứng: [tests](../outputs/extraction/verification_tests.txt), [summary](../outputs/extraction/verification_summary.json),
  [resume](../outputs/extraction/verification_resume.txt). Pilot 12 frame là artifact riêng, không thay cache full.
- **31/10:** rút gọn/tách tài liệu; gom code/config/tests vào `extraction/`, kết quả vào `outputs/extraction/`, Markdown vào `README/`.
  Sửa đường dẫn tài liệu trong code; thuật toán/config không đổi. Cache cũ giữ code hash gốc, nên
  resume bằng code mới sẽ yêu cầu output mới hoặc `--overwrite` có chủ đích.
- **Sau khi di chuyển:** 13 tests vẫn đạt; config mặc định và cả hai NPZ đọc được, nội dung NPZ không đổi.
  Log: [verification_layout_tests.txt](../outputs/extraction/verification_layout_tests.txt). Cấu trúc: [STRUCTURE.md](STRUCTURE.md).

**Chưa kiểm chứng:** toàn dataset và visual QA landmark mới. Chưa nối cache vào các
notebook đòi đặc trưng đã normalize; chưa có adapter gán bên tay. Chưa tạo venv riêng
hoặc đóng băng toàn bộ dependency/model binaries; không cam kết tái lập bitwise giữa máy.

## Cơ sở thiết kế

Tham khảo việc tách skeleton representation và preprocessing trong
[SPOTER–MediaPipe, mục 4](https://arxiv.org/html/2210.00893v1),
[SAM-SLR, mục 3.1/4.3](https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/papers/Jiang_Skeleton_Aware_Multi-Modal_Sign_Language_Recognition_CVPRW_2021_paper.html)
và [Roh et al., mục 3/4.2](https://aclanthology.org/2024.signlang-1.36.pdf).
75 điểm và NaN là lựa chọn của dự án. Quy ước/index tham chiếu:
[MediaPipe Pose](https://chuoling.github.io/mediapipe/solutions/pose.html),
[MediaPipe Hands](https://chuoling.github.io/mediapipe/solutions/hands.html).
