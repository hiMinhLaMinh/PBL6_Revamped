# Cấu trúc workspace

Cập nhật: 31/10/2026. Mỗi bước có thư mục mã nguồn riêng; các thư mục chưa triển khai để trống.

```text
PBL6_Revamped/
├── README/                    # Tài liệu từng bước/chủ đề đã làm
├── dataset/
│   ├── raw/                   # Video/dữ liệu gốc, giữ bất biến
│   ├── metadata/              # Nhãn, người ký, view, manifest nguồn
│   └── splits/                # Danh sách train/validation/test
├── extraction/                # Extractor, config, tests, requirements hiện có
├── preprocessing/             # Chuẩn hóa, sửa thiếu, xử lý thời gian
├── augmentation/              # Biến đổi dữ liệu phục vụ training
├── training/                  # Model, dataloader, vòng huấn luyện
├── evaluation/                # Đánh giá, metrics, phân tích lỗi
├── deployment/
│   ├── raspberry_pi/          # Ứng dụng/runtime trên Pi
│   └── server/                # Server extraction/inference nếu áp dụng
├── experiments/               # Cấu hình và mô tả thí nghiệm/ablation
├── notebooks/                 # Khảo sát, phân tích, trực quan tương tác
├── models/
│   ├── pretrained/            # Weights đầu vào
│   ├── checkpoints/           # Checkpoint huấn luyện
│   └── exported/              # Model xuất cho triển khai
└── outputs/
    ├── extraction/            # Raw NPZ và log kiểm chứng hiện có
    ├── preprocessing/         # Dữ liệu/đặc trưng đã xử lý
    ├── training/              # Log, lịch sử loss/metrics
    ├── evaluation/            # Bảng kết quả, biểu đồ, dự đoán
    └── deployment/            # Log và benchmark phần cứng/server
```

## Quy ước

- `dataset/raw/vsl400_input/` hiện có 7 Part; giữ video và metadata JSON gốc cùng cấu trúc nguồn.
  `dataset/metadata/` dành cho bảng tổng hợp/chuẩn hóa, không buộc tách metadata gốc khỏi Part.
- Mã/config/tests/dependencies hiện giữ cùng bước; kết quả sinh ra ghi vào `outputs/`.
- Weights thuộc `models/`; log và bảng số liệu thuộc `outputs/`. Khi có nhiều lần chạy,
  dùng thư mục con theo run ID và lưu config/seed/version cùng kết quả.
- `README/` chỉ có tài liệu cho phần đã làm hoặc vấn đề đã thảo luận. Mỗi file ghi
  quyết định, lý do và tiến trình ngắn; không tạo trước tài liệu cho mọi bước.
- Thư mục chưa có nội dung để trống thật, không thêm `.gitkeep`, file mẫu hoặc README giữ chỗ.

## Nội dung đã có

[Extraction](EXTRACTION.md) · [Association](HAND_ASSOCIATION.md) · [QC](EXTRACTION_QC.md) · [Mac/front](EXTRACTION_MAC.md) ·
[Tốc độ Pi 4](EXTRACTION_SPEED_PI4.md) · [Vision/cloud](EXTRACTION_VISION_CLOUD.md).

**Tiến trình 31/10:** tạo khung thư mục; chuyển `extraction/output/` sang `outputs/extraction/`
và sửa đường dẫn tài liệu. Giữ nguyên code/config và nội dung NPZ; chưa triển khai bước mới.
