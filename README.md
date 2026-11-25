# SignFlow - sửa nhận diện landmark A-Z

## Vì sao cần build và train lại?
Checkpoint landmark cũ chỉ có 18 lớp (J-Z và space), nên không thể dự đoán A-I.
Bản sửa loại `nothing` khỏi bài toán phân loại: `nothing` là trạng thái không có tay,
được webcam hiển thị riêng là `NO HAND DETECTED`.

## Cài thư viện
Kích hoạt môi trường `.venv311` của dự án, sau đó:
```powershell
python -m pip install opencv-python mediapipe==0.10.21 numpy tqdm
```
PyTorch đã cài trong môi trường thì không cần cài lại.

## Chép mã nguồn
Giải nén thư mục `src` trong gói này và thay thế thư mục `src` hiện tại của dự án.
Không xóa `models/best_model.pth`. Model CNN đó không bị thay đổi.

## 1. Tạo dữ liệu landmark
Chạy từ thư mục gốc dự án:
```powershell
python -m src.build_dataset --dataset "C:\Dev\nhan_dien_cu_chi_tay\dataset\asl_alphabet_train\asl_alphabet_train" --output "data\landmarks.npz"
```
Lệnh này xử lý toàn bộ ảnh mỗi lớp. Nó bỏ qua thư mục `nothing` và yêu cầu ít nhất
2 ảnh trích xuất được cho từng lớp còn lại. Nếu M hoặc N có tỉ lệ phát hiện thấp,
hãy xem báo cáo của hai lớp; không hạ ngưỡng một cách mù quáng.

Có thể chạy thử nhanh trước (100 ảnh/lớp):
```powershell
python -m src.build_dataset --dataset "C:\Dev\nhan_dien_cu_chi_tay\dataset\asl_alphabet_train\asl_alphabet_train" --output "data\landmarks_test.npz" --max-images-per-class 100
```

## 2. Huấn luyện model mới
Chỉ chạy sau khi bước 1 kết thúc thành công:
```powershell
python -m src.train --data "data\landmarks.npz" --output "models\landmark_model.pth" --epochs 80
```
Lệnh này ghi checkpoint landmark mới vào `models/landmark_model.pth`.
Nó không ghi đè `models/best_model.pth`.

## 3. Chạy webcam
```powershell
python -m src.webcam --model "models\landmark_model.pth" --threshold 0.65
```
Khi không phát hiện tay, webcam hiện `NO HAND DETECTED`.
Khi có tay nhưng độ tin cậy thấp, webcam hiện `UNKNOWN`.

## Lưu ý kiểm tra
- Khi khởi động, predictor phải in ra `Number of classes: 28` và danh sách A-Z, del, space.
- Nếu báo checkpoint không đủ 28 lớp, bạn đang dùng model cũ hoặc chưa train lại.
- `M` và `N` trong lần test trước có tỉ lệ MediaPipe phát hiện thấp; kết quả cuối phụ thuộc vào số mẫu thật sự trích xuất được và chất lượng ảnh.
