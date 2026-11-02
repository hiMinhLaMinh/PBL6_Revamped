# Extraction front-only trên Mac M3

Cập nhật: 01/11/2026. Máy đo thực tế: Apple M3 Pro, 11 core vật lý, RAM 18 GiB.

## Môi trường riêng đã kiểm chứng

Đã tạo `.venv/` tại project, CPython 3.11.9 arm64; không phụ thuộc venv PBL6 cũ.
`pip check` đạt; chỉ có một distribution OpenCV (`opencv-contrib-python`).
[Lock dependency](../extraction/requirements-macos-arm64.lock.txt) dành cho Python 3.11,
macOS arm64; muốn tái tạo trên máy phù hợp:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r extraction/requirements-macos-arm64.lock.txt
```

Dùng `.venv/bin/python` cho mọi lệnh dưới đây; không cần activate nếu gọi đường dẫn này.
`.venv/` đã được gitignore. File requirements.txt giữ dependency trực tiếp; lock pin cả phụ thuộc.

## Tối ưu giữ nguyên extraction condition

- Chỉ `front`: **24.753 video / 7 Part**, thay vì 74.259 video của ba view.
- [run_front_batch.py](../extraction/run_front_batch.py) khởi chạy mỗi Part trong Python
  process riêng; dùng tối đa N process đồng thời. Mỗi Part có manifest/lỗi/output riêng,
  không chia sẻ tracker hay writer CSV. Các thread chỉ điều phối subprocess, không fork graph.
- Giữ nguyên model/config/ảnh/frame. Chưa giảm complexity/resolution/FPS để tránh đổi
  chất lượng dữ liệu chỉ nhằm tăng tốc. Giảm số joint lưu không giảm phép tính mạng Pose.
- Giới hạn các pool OpenMP/BLAS/OpenCV của child về 1; chúng không phải XNNPACK thread setting.
  Lõi MediaPipe hiện dùng CPU/XNNPACK; có log GL/Metal không chứng minh neural inference trên GPU.

## Benchmark pilot v2 (lịch sử)

Cùng 14 clip (2 clip đầu mỗi Part), đầy đủ frame, cùng config và extraction code:

| Process đồng thời | Thời gian Part jobs | So với 1 process |
|---|---:|---:|
| 1 | 31.25 s | 1.00× |
| 2 | 18.25 s | 1.71× |
| 4 | 11.17 s | 2.80× |

Cả ba lần: 9 clip complete (567 frame), 5 clip failed với cùng identity. Các mảng NPZ
complete đối chiếu giống nhau, max difference=0; không coi lỗi/clip bị bỏ qua là speed-up.
Đây là pilot nhỏ, một lượt theo thứ tự 1→2→4; warm cache/nhiệt/nguồn điện chưa kiểm soát.
Chưa suy rộng ETA hoặc khẳng định 4 process tối ưu cho chạy dài. Khi đo lại, lặp và đảo thứ tự.

**4 process là ứng viên để đo dài hơn**. V3 đã xử lý overflow và chạy lại đủ 14/14 clip
trong 13.97 s Part jobs; không so tốc độ trực tiếp với v2 vì v2 dừng sớm 5 clip lỗi.
Review QC và hiệu chuẩn association trước khi khóa dữ liệu.
Duy trì nguồn điện, tránh Low Power Mode và cạnh tranh CPU, theo dõi RAM/swap/nhiệt khi chạy dài.

## Cách chạy

Từ project root, preflight toàn bộ input:

```bash
.venv/bin/python extraction/run_front_batch.py dataset/raw/vsl400_input \
  --output outputs/extraction/front_v3 --workers 4 --dry-run
```

Pilot: thay `--dry-run` bằng `--max-videos-per-part 2`, chọn output pilot riêng.
Thu thập raw đầy đủ: bỏ giới hạn pilot, thay `--dry-run` bằng `--resume`; có thể thêm
`caffeinate -i` trước lệnh để ngăn idle sleep. Association có thể hiệu chỉnh offline sau đó.
Giữ nguyên runner/input root qua các lần resume: NPZ ở `front_v3/Part_N/front_view/`;
manifest/lỗi ở từng `Part_N/`, log/run summary ở `_batch/<run_id>/`.
Không chạy hai runner đồng thời vào cùng output. Resume không đổi config/model để chữa lỗi.

[Quality control](EXTRACTION_QC.md) chạy sau extraction, đọc cả manifest và NPZ.
Runner chỉ điều phối các extractor giống nhau; `--association-policy` truyền policy cho mọi Part.
Raw v3 giữ candidates và association L/R riêng để chạy lại association offline.

## Đánh giá sẵn sàng (01/11)

Trong venv mới: **55 tests đạt; preflight 24.753 front / 7 Part, 0 metadata invalid**.
Pilot 10 clip đầu/Part: **70/70 complete, 4.635 frame**, 18 frame overflow được giữ;
resume skip đúng 70 cache, cả 14 clip pilot cũ có mảng bằng nhau hoàn toàn.
QC: 43 not_flagged, 27 review, 0 batch lỗi; review không đồng nghĩa phải loại clip.
Đây là 7 signer, chưa đại diện mọi signer/gloss, chưa là kiểm chứng chạy liên tục nhiều giờ.

**Sẵn sàng thu raw hàng loạt**, chưa chứng nhận quality cho training. Giữ cấu hình detector
ổn định; hậu kiểm coverage/incomplete, review trực quan và hiệu chuẩn association/QC trước training.
Ước NPZ ~4.1 GiB theo bytes/frame của pilot, chưa gồm log/preview; hiện trống ~93 GiB.
[Báo cáo môi trường/pilot](../outputs/extraction/venv_readiness_verification/summary.json).
Chưa khởi chạy full batch.

## Bằng chứng và nguồn

[Benchmark](../outputs/extraction/mac_qc_verification/benchmark_runs.json),
[dry-run front](../outputs/extraction/mac_qc_verification/dry_run_front.json).
[Pilot v3](../outputs/extraction/association_v3_verification/summary.json): 14/14 complete,
9 frame overflow được giữ, 5 clip review. Chưa chạy toàn bộ dataset.

Nguồn: [MediaPipe 0.10.14 thread defaults](https://github.com/google-ai-edge/mediapipe/blob/v0.10.14/mediapipe/calculators/tensor/inference_calculator_utils.cc),
[Python multiprocessing](https://docs.python.org/3/library/multiprocessing.html),
[LiteRT profiling/thread guidance](https://developers.google.com/edge/litert/conversion/tensorflow/build/best_practices).
