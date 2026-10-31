# Tăng tốc trích xuất landmark trên Raspberry Pi 4B

Cập nhật: 31/10/2026. **Trạng thái: phân tích và đề xuất; chưa triển khai hoặc benchmark trên Pi.**
Extractor raw đang dùng được mô tả tại [tài liệu trích xuất](EXTRACTION.md).

## Bài toán và ràng buộc

- Pi 4B 8GB hiện có; không bổ sung phần cứng. Bottleneck là MediaPipe, không phải classifier.
- Người dùng báo khoảng **5.x FPS** khi Pose và Hands đều complexity=0; chưa đo lại bằng Revamped.
- Có thể thay model và thiết kế pipeline mới. Config raw hiện tại vẫn complexity=1;
  thảo luận triển khai chưa thay cấu hình hoặc yêu cầu raw 75 điểm.
- Ví dụ 5.5 FPS ≈ 182 ms/frame; đủ 25 FPS cần trung bình ≤40 ms/frame, còn phải xét p95/reacquire.

## Những gì đã xác minh

| Phát hiện | Hệ quả cần thử |
|---|---|
| Graph 0.10.14 trên máy phát triển có `xnnpack {}` ở bốn inference nodes, chưa đặt threads. Mã nguồn mặc định trả 1 thread ở nhánh Linux desktop. | Benchmark explicit 1/2/3/4 XNNPACK threads trên đúng bản cài Pi. `cpu_num_thread` và `OMP_NUM_THREADS` không thay thế cấu hình delegate. |
| Hands gọi palm detector khi số tay đang track chưa đạt `max_num_hands`. | Với cấu hình 2 tay, đoạn chỉ có 1 tay có thể tốn thêm detection. Đo riêng 0/1/2 tay và lịch tìm tay mới. |
| Landmark subgraph vốn crop/resize ROI tay về **224×224**. | Resize ảnh camera không tự giảm kích thước mạng landmark; bỏ Z/bớt joint ở output cũng không giảm đáng kể chi phí chạy mạng. |

Đây là kiểm tra source/graph trên máy phát triển, chưa xác nhận build hoặc mức tăng FPS trên Pi.

## Hướng ưu tiên: detector nhẹ → landmark-only

```text
Hand detector ở startup / mất track / chu kỳ tìm tay mới
  → ROI tracker từng tay → crop có padding và căn hướng
  → MediaPipe hand landmark lite → chiếu điểm về ảnh gốc
```

**Cần thay palm detector nội bộ:** dùng `HandLandmarkCpu` (nhận IMAGE + ROI `NormalizedRect`)
hoặc gọi landmark TFLite riêng. Crop rồi gọi `Hands.process(crop)` vẫn giữ cả pipeline
detection/tracking, có thể thành hai lượt detection. Nếu thử cách đó với hai crop video,
phải tách tracking state; không luân phiên crop vào cùng một tracker.

Cần giữ đủ ngón/cổ tay, chuyển bbox sang ROI phù hợp, inverse transform và giữ identity
hai tay khi giao nhau/che khuất. Detector chỉ chạy thưa khi tracking đáng tin; đo độ trễ
phát hiện tay mới. Khi MediaPipe đang track tốt đủ hai tay, palm detector vốn đã được skip;
detector ngoài chạy mỗi frame có thể làm chậm hơn. Chưa chọn detector cụ thể.

Chi phí tuần tự xấp xỉ:

`T = T_detector/K + n_hands × T_landmark + T_crop/tracking/postprocess + T_body_nếu_giữ`

Nếu riêng landmark hai tay đã vượt budget, thay detector chưa đủ để đạt 25 FPS.
Hand detector cũng không thay thế body landmarks nếu model nhận dạng vẫn cần chúng.

## Các hướng bổ sung

| Hướng | Lưu ý / đánh đổi |
|---|---|
| Explicit XNNPACK threads; TFLite trực tiếp hoặc ncnn ARM64 | Thử trước khi train model mới; tránh tranh CPU. MediaPipe vốn chạy neural inference native, đổi vòng Python sang C++ không bảo đảm tăng tốc. |
| Body chạy thưa hơn Hands; thử MoveNet Lightning INT8 | MoveNet có 17 điểm, input 192×192; cần adapter/topology mới, không cung cấp 21 điểm ngón tay. Anchor có thể cũ khi người ký di chuyển. |
| Hand regressor nhỏ hơn, input 96/128, output 21 XY, distillation/QAT | Cần model được train/chuyển đổi phù hợp; không chỉ resize input model hiện tại. Kiểm tra sai số ngón tay và nhãn teacher. |
| Optical flow giữa các frame neural | Có thể drift khi blur/che khuất/đổi hình tay. Output 25 Hz không đồng nghĩa 25 lượt neural landmark mới. |
| Buffer frame mới nhất, tracking/video mode, giảm copy/overlay | Giảm backlog/overhead; không làm một lượt inference vốn chậm trở nên nhanh hơn. |

Đổi sang 7 điểm thân + 42 điểm tay XY là một đề xuất deployment, **chưa chốt**.
Extractor mới cần schema/identity riêng; không giả tạo joint/Z để giống cache MediaPipe 75 điểm.
Tài liệu này tập trung chạy local trên Pi; hướng offload được phân tích riêng tại
[Vision API/cloud](EXTRACTION_VISION_CLOUD.md). Không đề xuất mua phần cứng mới.

## Benchmark để quyết định

1. Hands+Pose hiện tại và Hands-only: tách chi phí body, ghi số tay/lượt detection.
2. Hai ROI tay chuẩn → landmark-only: xác định giới hạn trước khi thêm detector mới.
3. Detector ngoài + landmark-only + tracker: đo toàn pipeline và thời gian reacquire.

Dùng cùng clip/input, đo sau warm-up và khi chạy liên tục: timing từng mạng/stage,
FPS, capture-to-result p50/p95, dropout, accuracy ISLR và throttling. Demo ncnn công bố
khoảng 7 FPS trên Pi 4 chỉ là tham khảo, không chứng minh pipeline này sẽ đạt 25 FPS.

Nếu observation rate thấp hơn training, đánh giá lại sampling theo timestamp thực tế.
Nội suy 10 quan sát lên 64 không phục hồi chuyển động ngón tay bị mất; subsample cache
25 FPS cũng chưa mô phỏng đầy đủ việc tracker nhận ảnh thưa hơn.

## Tiến trình và nguồn

- **30/10:** phân tích báo cáo 5.x FPS; kiểm tra graph/thread defaults và detector gating;
  thảo luận detector nhẹ + crop, landmark-only, runtime và model thay thế.
- **31/10:** gom nội dung hiệu năng vào tài liệu này. Chưa chọn backend mới hoặc đo speed-up trên Pi.

Nguồn: MediaPipe 0.10.14
[thread selection](https://github.com/google-ai-edge/mediapipe/blob/v0.10.14/mediapipe/calculators/tensor/inference_calculator_utils.cc),
[inference options](https://github.com/google-ai-edge/mediapipe/blob/v0.10.14/mediapipe/calculators/tensor/inference_calculator.proto),
[Hands tracking](https://github.com/google-ai-edge/mediapipe/blob/v0.10.14/mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.pbtxt),
[landmark ROI](https://github.com/google-ai-edge/mediapipe/blob/v0.10.14/mediapipe/modules/hand_landmark/hand_landmark_cpu.pbtxt).
Tham khảo thêm:
[MoveNet](https://www.tensorflow.org/hub/tutorials/movenet),
[ncnn](https://github.com/Tencent/ncnn),
[demo hand pose Pi 4](https://github.com/Qengineering/Hand-Pose-ncnn-Raspberry-Pi-4),
[LiteRT performance](https://developers.google.com/edge/litert/conversion/tensorflow/build/best_practices),
[OpenCV optical flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html),
[Tasks live-stream behavior](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python).
