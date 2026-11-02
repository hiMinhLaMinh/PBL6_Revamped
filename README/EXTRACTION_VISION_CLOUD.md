# Trích xuất bằng Apple Vision hoặc cloud

Cập nhật: 31/10/2026. **Trạng thái: phân tích, chưa triển khai/gửi video lên dịch vụ.**
[Tài liệu extraction hiện tại](EXTRACTION.md) · [Tối ưu local Pi 4](EXTRACTION_SPEED_PI4.md).

## Phân biệt các lựa chọn

| Hướng | Khả năng và mức phù hợp |
|---|---|
| **Apple Vision** | Framework trên nền tảng Apple; hand pose có 21 điểm 2D/tay và confidence. Không chạy native trên Pi/Linux; có thể dùng Mac hiện có làm máy extraction offline hoặc server LAN. |
| **Google Cloud Vision** | Danh sách tính năng không có bộ 21 joint tay; LANDMARK_DETECTION nhận diện địa danh. Không thay trực tiếp skeleton extractor ISLR. |
| **Google Video Intelligence: Person Detection** | Có 17 điểm cơ thể gồm wrist; không có các joint ngón tay trong danh sách output này. Không đủ cho ký hiệu cần hình dạng tay. |
| **Cloud tự phục vụ model** | Đưa MediaPipe hoặc model body/hand chuyên dụng lên VM/container/endpoint. Kiểm soát được skeleton và phiên bản, nhưng cần xây transport/session và trả phí compute/network/storage. |

## Apple Vision: phù hợp ở đâu?

- Dùng `VNDetectHumanHandPoseRequest` cho tối đa hai tay và body request riêng nếu cần.
  Hand API này trả 2D; không giả định có Z/world của MediaPipe. Chuẩn hóa joint mapping,
  confidence, ảnh mirror/orientation; Vision dùng gốc tọa độ dưới-trái, cần adapter nếu
  schema đích dùng trên-trái. Chưa có benchmark Apple Vision trên máy hiện tại.
- Thử offline trên cùng clip trước, đo thời gian/frame, dropout hai tay, crossing/occlusion
  và chất lượng downstream. Nếu dùng live: Pi gửi ảnh/video sang Mac, Mac trả landmark.
  Đây là kiến trúc phân tán phụ thuộc Mac đang hoạt động, không còn Pi xử lý độc lập.
- Apple công bố đổi hand model năm 2025: vẫn 21 điểm nhưng vị trí ước lượng có thay đổi;
  khuyến nghị train lại classifier. Ghi macOS/build/request revision và extraction condition;
  không coi Vision mới, Vision cũ và MediaPipe là dữ liệu tương đương chỉ vì cùng số joint.
- Dùng Vision tạo training data rồi deploy MediaPipe trên Pi cần đánh giá chênh lệch
  phân bố extractor. Nếu theo hướng Vision, ưu tiên train/evaluate với đúng backend dự kiến.

## Cloud: kiến trúc nên thử

```text
Pi: capture + encode -> kết nối stream -> server: decode + skeleton extractor
                    <- frame_id + timestamp + landmark/confidence/schema <-
Pi: preprocessing + classifier hiện có + UI
```

Giữ classifier trên Pi là một lựa chọn vì nó không phải bottleneck. Server có thể chạy
MediaPipe cùng phiên bản để giảm số thay đổi, hoặc thử model khác với schema riêng.
Nếu dùng tracking, giữ thứ tự frame và state riêng từng session; không trộn frame người
khác hoặc phân tán ngẫu nhiên mỗi frame qua workers. Mất kết nối cần khởi tạo lại state.

- **Dataset offline:** upload clip một lần, xử lý batch, tải NPZ về; giữ đủ frame/timestamp
  và hash nguồn/config/model. Hợp với mục tiêu raw foundation, không cần đạt real time.
- **Live:** ưu tiên kết nối lâu dài (WebSocket/gRPC/WebRTC tùy transport), queue có giới hạn,
  frame ID/timestamp và backpressure; tránh mở một kết nối HTTP mới cho mỗi frame.
  Endpoint serverless có thể có cold start; cân nhắc instance giữ nóng và chi phí tương ứng.
- Độ trễ tổng = capture/encode + upload + queue + decode/extract + download + xử lý phía Pi.
  Server nhanh hơn không tự đảm bảo latency thấp hơn. Đo p50/p95 cả đường truyền.
- **Throughput khác latency:** có thể pipeline nhiều frame và trả 25 kết quả/giây dù mỗi
  kết quả trễ hơn 40 ms; gửi một frame rồi chờ response mới gửi frame sau sẽ bị RTT giới hạn.
- Ví dụ minh họa: JPEG 50 KB × 25 FPS ≈ **10 Mbps upload**, chưa overhead; không phải số
  đo thực tế. Stream video có thể giảm băng thông nhưng cần kiểm soát buffering/codec/độ nét tay.
- Chi phí model tự host tính theo instance time, network và lưu trữ; API tính theo frame
  (nếu có) sẽ nhận tới 90.000 frame/giờ ở 25 FPS. Chưa chọn nhà cung cấp hoặc báo giá.
  Cần chốt video có lưu ở server hay chỉ xử lý tạm trước khi triển khai.

## Đề xuất và bước kiểm chứng tiếp theo

1. Nếu cần **Pi độc lập/offline**, tiếp tục hướng local; Apple/cloud không thỏa điều kiện đó.
2. Nếu chấp nhận server, ưu tiên thử **cùng extractor trên máy hiện có qua LAN** để đo lợi ích
   của offload trước khi trả phí cloud; thêm Apple Vision như nhánh so sánh riêng nếu muốn.
3. Nếu cần truy cập từ xa, benchmark custom extractor tại region phù hợp: chất lượng landmark,
   accuracy ISLR, FPS, latency p50/p95, jitter, băng thông, reconnect và chi phí/giờ.

Chỉ là thứ tự đề xuất; chưa triển khai server hoặc đo speed-up. Raw local đã đổi sang representation 50 điểm/schema v3, giữ thêm mọi hand candidate ngày 01/11.
**Tiến trình 31/10:** đọc nguồn chính thức Apple/Google, phân biệt API có sẵn và custom
serving, ghi các đánh đổi; tổ chức lại tài liệu theo từng chủ đề.

## Nguồn

- [Apple: body/hand pose và tọa độ/confidence](https://developer.apple.com/videos/play/wwdc2020/10653/).
- [Apple: hand model cập nhật, WWDC25](https://developer.apple.com/videos/play/wwdc2025/272/).
- [Cloud Vision: danh sách tính năng](https://docs.cloud.google.com/vision/docs/features-list),
  [Video Intelligence: body landmarks](https://docs.cloud.google.com/video-intelligence/docs/feature-person-detection).
- [Custom model serving/container](https://docs.cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements),
  [Cloud Run WebSockets](https://docs.cloud.google.com/run/docs/triggering/websockets).
