# SignFlow – Real-Time Hand Gesture Recognition

**SignFlow** is a Computer Vision and Machine Learning project designed to recognize American Sign Language (ASL) alphabet hand gestures in real time using a webcam.

The project combines **MediaPipe Hands**, **PyTorch**, and **OpenCV** to detect hand landmarks, extract geometric features, classify hand gestures, and display recognition results directly from a live video stream.

The system focuses on recognizing hand shapes rather than classifying raw images directly. By extracting and learning the spatial coordinates of 21 hand landmarks, SignFlow provides a lightweight approach to hand gesture classification.

---

## 1. Project Overview

Hand gestures are an important form of communication, particularly for people with hearing or speech impairments. However, communication barriers can arise when other people do not understand sign language.

SignFlow aims to explore how Computer Vision and Deep Learning can help bridge this communication gap through automated hand gesture recognition.

The system recognizes 28 gesture classes:

- 26 alphabet letters: A–Z.
- `del`: Delete operation.
- `space`: Space operation.

The `nothing` category in the original dataset is excluded from model training because it represents the absence of a detected hand rather than an actual gesture. Instead, the application handles this condition directly during webcam inference.

## 2. Project Objectives

The main objectives of SignFlow are:

- Develop a real-time hand gesture recognition system using Computer Vision.
- Detect and extract 21 hand landmarks using MediaPipe.
- Transform hand landmarks into numerical feature vectors for machine learning.
- Train a PyTorch-based classification model to recognize different hand gestures.
- Integrate the trained model with a webcam for real-time prediction.
- Improve recognition stability through confidence thresholding and prediction smoothing.
- Explore practical applications of sign language recognition in everyday communication.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Image processing and webcam capture |
| MediaPipe Hands | Hand detection and landmark extraction |
| PyTorch | Neural network development and training |
| NumPy | Numerical computation and feature processing |
| tqdm | Training and dataset processing progress |

## 4. System Architecture

SignFlow follows a landmark-based recognition pipeline.

```text
             Input Image / Webcam
                      |
                      v
               Hand Detection
                 MediaPipe
                      |
                      v
             Extract 21 Landmarks
                 (x, y, z)
                      |
                      v
              Landmark Normalization
                      |
                      v
             Feature Vector (63D)
                      |
                      v
             PyTorch Classifier
                      |
                      v
              Softmax Prediction
                      |
                      v
             Confidence Threshold
                      |
                      v
              Prediction Smoothing
                      |
                      v
              Recognized Gesture
                      |
                      v
               Webcam Display
```

### 4.1. Hand Landmark Detection

MediaPipe Hands detects 21 key points on a hand, including:

- Wrist.
- Thumb joints.
- Index finger joints.
- Middle finger joints.
- Ring finger joints.
- Little finger joints.

Each landmark contains three coordinates: X, Y, and Z.

Therefore, each detected hand is represented by:

**21 landmarks × 3 coordinates = 63 numerical features.**

### 4.2. Landmark Normalization

To reduce the influence of hand position and scale, the extracted coordinates are normalized before being passed to the classifier.

The normalization process includes:

- Moving the wrist landmark to the origin.
- Scaling coordinates based on the distance between the wrist and the middle finger's MCP joint.
- Flattening the normalized coordinates into a 63-dimensional feature vector.

The same normalization procedure is applied during dataset preparation and webcam inference to maintain consistency.

### 4.3. Deep Learning Classification

SignFlow uses a Multi-Layer Perceptron (MLP) implemented with PyTorch.

The model receives 63 input features and learns to classify them into the corresponding gesture classes.

The network architecture consists of:

| Layer | Configuration |
|---|---|
| Input | 63 features |
| Hidden Layer 1 | 256 neurons |
| Hidden Layer 2 | 128 neurons |
| Hidden Layer 3 | 64 neurons |
| Output | 28 classes |

The network uses ReLU activation, Layer Normalization, Dropout, and a final classification layer.

### 4.4. Real-Time Recognition

During webcam inference, the system continuously:

1. Captures frames from the webcam using OpenCV.
2. Detects the hand using MediaPipe.
3. Extracts and normalizes 21 landmarks.
4. Generates a 63-dimensional feature vector.
5. Predicts the corresponding gesture using the trained PyTorch model.
6. Evaluates prediction confidence.
7. Applies temporal prediction smoothing to reduce unstable results.
8. Displays the predicted gesture and confidence score.

If no hand is detected, the application displays `NO HAND DETECTED`.

If the prediction confidence is below the configured threshold, the application displays `UNKNOWN`.

---

## 5. Dataset

SignFlow uses an ASL Alphabet image dataset organized into folders according to gesture labels.

The dataset contains the following gesture categories:

- Letters A–Z.
- Delete (`del`).
- Space (`space`).
- Nothing (`nothing`).

During dataset preparation, the project extracts hand landmarks from the original images and converts them into numerical feature vectors.

The resulting dataset is stored in compressed NumPy format (`.npz`), containing:

- `X`: Extracted landmark features.
- `y`: Numerical class labels.
- `class_names`: Corresponding gesture class names.

The `nothing` category is excluded because the absence of a hand is handled separately by the webcam application.

## 6. Project Structure

```text
SignFlow/
│
├── data/
│   └── landmarks.npz
│
├── dataset/
│   └── ASL Alphabet Dataset
│
├── models/
│   └── landmark_model.pth
│
├── notebooks/
│   └── Model training and experimentation
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── landmarks.py
│   ├── model.py
│   ├── build_dataset.py
│   ├── train.py
│   ├── predictor.py
│   └── webcam.py
│
├── requirements.txt
└── README.md
```

*Note: The directory structure above describes the main components of the project. Dataset and model files may be stored locally and are not necessarily included in the Git repository.*

## 7. Installation

### Requirements

- Python 3.10 or a compatible Python version.
- Webcam.
- Windows, Linux, or macOS.
- PyTorch.
- MediaPipe 0.10.21.

### Step 1. Clone the repository

```bash
git clone https://github.com/khoi27122007-tech/SignFlow.git
cd SignFlow
```

### Step 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

### Step 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

Make sure the installed PyTorch version supports your hardware configuration if GPU acceleration is required.

---

## 8. Running the Project

### Step 1. Prepare the landmark dataset

Run the following command from the project root directory:

```bash
python -m src.build_dataset --dataset "path/to/ASL_Dataset" --output "data/landmarks.npz"
```

This process detects hands in the dataset images, extracts 21 landmarks, normalizes the coordinates, and generates the feature dataset.

### Step 2. Train the classification model

```bash
python -m src.train --data "data/landmarks.npz" --output "models/landmark_model.pth" --epochs 80
```

The training process includes:

- Dataset validation.
- Stratified training and validation splitting.
- Model optimization.
- Validation loss and accuracy evaluation.
- Saving the best model checkpoint.

### Step 3. Run real-time webcam recognition

```bash
python -m src.webcam --model "models/landmark_model.pth" --threshold 0.65
```

The webcam application will start detecting and recognizing hand gestures in real time.

Press `Q` or `ESC` to close the application.

### Optional parameters

| Parameter | Description |
|---|---|
| `--model` | Path to the trained model |
| `--camera` | Webcam device index |
| `--threshold` | Minimum prediction confidence |
| `--smooth-window` | Number of recent predictions used for smoothing |
| `--no-mirror` | Disable mirrored webcam display |

---

## 9. Real-World Applications

SignFlow explores several potential applications of automated hand gesture recognition.

### 9.1. Supporting Communication for People with Hearing and Speech Impairments

A gesture recognition system can help convert recognized hand gestures into text, making communication more accessible between people who understand sign language and those who do not.

### 9.2. Accessibility Applications

The recognition technology can serve as a foundation for accessibility software, allowing users to interact with digital devices through hand gestures.

### 9.3. Smart Human-Computer Interaction

Hand gesture recognition can be integrated into interactive applications to provide alternative input methods without requiring a traditional keyboard or mouse.

### 9.4. Educational Tools

SignFlow can be extended into an educational application that helps learners practice ASL alphabet gestures and receive immediate feedback on their hand shapes.

### 9.5. Research and Development

The project provides a practical foundation for further research in:

- Computer Vision.
- Hand pose estimation.
- Landmark-based classification.
- Human-Computer Interaction.
- Real-time gesture recognition.
- Assistive technologies.

**Important:** The current implementation recognizes isolated ASL alphabet gestures. It does not yet translate continuous sign language sentences, interpret facial expressions, or understand full sign language grammar.

---

## 10. Current Limitations

Although SignFlow provides real-time hand gesture classification, several limitations remain:

- Recognition performance depends on hand visibility and MediaPipe detection quality.
- Similar hand shapes may produce incorrect predictions.
- Lighting conditions, camera quality, and viewing angles can affect detection.
- The current model is designed for one detected hand.
- The current implementation recognizes individual static gestures rather than continuous sign language sequences.
- Recognition accuracy depends on the quality, diversity, and distribution of training data.

---

## 11. Future Improvements

Potential future developments include:

- Expanding recognition to more sign language gestures.
- Supporting two-hand gesture recognition.
- Improving robustness under different lighting and camera conditions.
- Building a graphical user interface.
- Converting recognized gestures into continuous text.
- Adding text-to-speech functionality.
- Exploring dynamic gesture recognition using temporal models.
- Deploying the system as a desktop or web application.

---

## 12. License and Contributions

This project is intended for educational, research, and experimental purposes.

Contributions, suggestions, and improvements are welcome.

If you find an issue or have an idea for improving SignFlow, feel free to open an issue or submit a pull request.

---

**SignFlow – Exploring Computer Vision for More Accessible Communication.**

# SignFlow – Hệ thống nhận diện cử chỉ bàn tay theo thời gian thực

**SignFlow** là dự án ứng dụng Thị giác máy tính (Computer Vision) và Học máy (Machine Learning) nhằm xây dựng hệ thống nhận diện các cử chỉ bàn tay thuộc bảng chữ cái Ngôn ngữ ký hiệu Mỹ (American Sign Language – ASL) thông qua webcam theo thời gian thực.

Dự án kết hợp **MediaPipe Hands, PyTorch và OpenCV** để phát hiện bàn tay, trích xuất các điểm mốc đặc trưng, phân loại cử chỉ và hiển thị kết quả nhận diện trực tiếp trên màn hình.

Thay vì sử dụng trực tiếp toàn bộ hình ảnh làm đầu vào cho mô hình, SignFlow áp dụng phương pháp nhận diện dựa trên đặc trưng hình học của bàn tay thông qua 21 điểm mốc (Hand Landmarks). Phương pháp này giúp biểu diễn cấu trúc bàn tay dưới dạng dữ liệu số, từ đó phục vụ quá trình huấn luyện và dự đoán cử chỉ.

---

## 1. Giới thiệu dự án

Ngôn ngữ ký hiệu là một phương thức giao tiếp quan trọng đối với cộng đồng người khiếm thính và người gặp khó khăn trong giao tiếp bằng lời nói. Tuy nhiên, sự khác biệt về khả năng sử dụng ngôn ngữ ký hiệu có thể tạo ra rào cản trong quá trình giao tiếp với những người chưa có kiến thức về loại ngôn ngữ này.

SignFlow được xây dựng nhằm nghiên cứu và ứng dụng công nghệ nhận diện cử chỉ bàn tay tự động, góp phần tạo nền tảng cho các giải pháp hỗ trợ giao tiếp và tăng khả năng tiếp cận công nghệ.

Hệ thống hiện hỗ trợ nhận diện **28 lớp cử chỉ**, bao gồm:

- 26 chữ cái trong bảng chữ cái tiếng Anh: A–Z.
- `del`: Cử chỉ xóa.
- `space`: Cử chỉ tạo khoảng trắng.

Trong bộ dữ liệu gốc có thêm lớp `nothing`, đại diện cho trạng thái không có bàn tay. Lớp này không được đưa vào quá trình huấn luyện vì hệ thống xử lý trạng thái không phát hiện được bàn tay trực tiếp trong quá trình chạy webcam.

## 2. Mục đích xây dựng dự án

SignFlow được phát triển với các mục tiêu chính:

- Nghiên cứu và ứng dụng Computer Vision trong bài toán nhận diện cử chỉ bàn tay.
- Sử dụng MediaPipe để phát hiện và trích xuất 21 điểm mốc trên bàn tay.
- Chuyển đổi dữ liệu tọa độ thành các vector đặc trưng phục vụ huấn luyện mô hình.
- Xây dựng mô hình phân loại cử chỉ bằng PyTorch.
- Tích hợp mô hình đã huấn luyện với webcam để nhận diện theo thời gian thực.
- Cải thiện độ ổn định của kết quả thông qua ngưỡng độ tin cậy và cơ chế làm mượt dự đoán.
- Tìm hiểu khả năng ứng dụng trí tuệ nhân tạo vào các giải pháp hỗ trợ giao tiếp và tương tác giữa con người với máy tính.

## 3. Công nghệ sử dụng

| Công nghệ | Vai trò |
|---|---|
| Python | Ngôn ngữ lập trình chính |
| OpenCV | Xử lý hình ảnh và thu nhận dữ liệu từ webcam |
| MediaPipe Hands | Phát hiện bàn tay và trích xuất điểm mốc |
| PyTorch | Xây dựng, huấn luyện và triển khai mô hình phân loại |
| NumPy | Xử lý dữ liệu số và vector đặc trưng |
| tqdm | Hiển thị tiến trình xử lý dữ liệu |

## 4. Kiến trúc hệ thống

SignFlow sử dụng kiến trúc nhận diện cử chỉ dựa trên đặc trưng điểm mốc bàn tay (Landmark-based Hand Gesture Recognition).

### 4.1. Quy trình hoạt động

```text
          Hình ảnh đầu vào / Webcam
                      |
                      v
              Phát hiện bàn tay
                  MediaPipe
                      |
                      v
           Trích xuất 21 điểm mốc
                  (x, y, z)
                      |
                      v
             Chuẩn hóa tọa độ
                      |
                      v
          Vector đặc trưng 63 chiều
                      |
                      v
           Mô hình phân loại PyTorch
                      |
                      v
             Dự đoán xác suất
                      |
                      v
           Kiểm tra độ tin cậy
                      |
                      v
            Làm mượt kết quả
                      |
                      v
           Hiển thị cử chỉ nhận diện
```

### 4.2. Phát hiện và trích xuất điểm mốc bàn tay

SignFlow sử dụng MediaPipe Hands để phát hiện bàn tay và xác định 21 điểm mốc quan trọng, bao gồm:

- Cổ tay (Wrist).
- Các khớp của ngón cái (Thumb).
- Các khớp của ngón trỏ (Index Finger).
- Các khớp của ngón giữa (Middle Finger).
- Các khớp của ngón áp út (Ring Finger).
- Các khớp của ngón út (Pinky Finger).

Mỗi điểm mốc chứa ba tọa độ không gian X, Y và Z.

Do đó, dữ liệu của một bàn tay được biểu diễn bằng:

**21 điểm mốc × 3 tọa độ = 63 đặc trưng.**

### 4.3. Chuẩn hóa dữ liệu

Sau khi trích xuất, các tọa độ được chuẩn hóa để giảm ảnh hưởng của vị trí và kích thước bàn tay trong khung hình.

Quy trình bao gồm:

- Đưa điểm cổ tay về vị trí gốc tọa độ.
- Chuẩn hóa tỷ lệ dựa trên khoảng cách từ cổ tay đến khớp MCP của ngón giữa.
- Chuyển đổi toàn bộ tọa độ thành vector đặc trưng 63 chiều.

Quá trình chuẩn hóa được sử dụng thống nhất trong cả giai đoạn xây dựng dữ liệu và nhận diện bằng webcam.

### 4.4. Xây dựng mô hình học sâu

Dự án sử dụng mạng nơ-ron truyền thẳng nhiều lớp (Multi-Layer Perceptron – MLP) được xây dựng bằng PyTorch.

Mô hình tiếp nhận 63 đặc trưng đầu vào và học cách phân loại chúng thành các lớp cử chỉ tương ứng.

| Thành phần | Cấu hình |
|---|---|
| Đầu vào | 63 đặc trưng |
| Hidden Layer 1 | 256 neuron |
| Hidden Layer 2 | 128 neuron |
| Hidden Layer 3 | 64 neuron |
| Đầu ra | 28 lớp |

Mạng sử dụng các kỹ thuật ReLU, Layer Normalization và Dropout nhằm hỗ trợ quá trình học và hạn chế hiện tượng quá khớp.

### 4.5. Nhận diện theo thời gian thực

Trong quá trình chạy webcam, hệ thống liên tục thực hiện:

1. Thu nhận khung hình bằng OpenCV.
2. Phát hiện bàn tay bằng MediaPipe.
3. Trích xuất và chuẩn hóa 21 điểm mốc.
4. Tạo vector đặc trưng gồm 63 giá trị.
5. Đưa dữ liệu vào mô hình PyTorch để dự đoán.
6. Kiểm tra độ tin cậy của kết quả.
7. Làm mượt dự đoán dựa trên các kết quả gần nhất.
8. Hiển thị cử chỉ và độ tin cậy trên màn hình.

Nếu không phát hiện được bàn tay, hệ thống hiển thị `NO HAND DETECTED`.

Nếu có bàn tay nhưng độ tin cậy dự đoán thấp hơn ngưỡng quy định, hệ thống hiển thị `UNKNOWN`.

## 5. Bộ dữ liệu (Dataset)

SignFlow sử dụng bộ dữ liệu hình ảnh ASL Alphabet được tổ chức theo từng thư mục tương ứng với nhãn cử chỉ.

Bộ dữ liệu bao gồm các nhóm:

- Các chữ cái từ A đến Z.
- Cử chỉ `del`.
- Cử chỉ `space`.
- Trạng thái `nothing`.

Trong quá trình xây dựng dữ liệu, hệ thống sử dụng MediaPipe để trích xuất đặc trưng bàn tay từ các hình ảnh gốc, sau đó chuyển đổi thành dữ liệu số phục vụ huấn luyện.

Dữ liệu sau xử lý được lưu dưới định dạng nén NumPy (`.npz`), bao gồm:

| Thành phần | Ý nghĩa |
|---|---|
| `X` | Ma trận đặc trưng điểm mốc bàn tay |
| `y` | Nhãn số tương ứng với từng mẫu |
| `class_names` | Danh sách tên các lớp cử chỉ |

Lớp `nothing` được loại khỏi quá trình huấn luyện vì trạng thái không có bàn tay được xử lý trực tiếp trong chương trình webcam.

## 6. Cấu trúc thư mục dự án

```text
SignFlow/
│
├── data/
│   └── landmarks.npz
│
├── dataset/
│   └── ASL Alphabet Dataset
│
├── models/
│   └── landmark_model.pth
│
├── notebooks/
│   └── Notebook thử nghiệm và huấn luyện
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── landmarks.py
│   ├── model.py
│   ├── build_dataset.py
│   ├── train.py
│   ├── predictor.py
│   └── webcam.py
│
├── requirements.txt
└── README.md
```

*Lưu ý: Đây là cấu trúc mô tả các thành phần chính của dự án. Các thư mục chứa dataset, dữ liệu đã xử lý và mô hình có thể được lưu cục bộ và không nhất thiết được tải lên GitHub.*

## 7. Hướng dẫn cài đặt

### Yêu cầu hệ thống

- Python 3.10 hoặc phiên bản tương thích.
- Webcam.
- Windows, Linux hoặc macOS.
- PyTorch.
- MediaPipe 0.10.21.

### Bước 1. Tải dự án

```bash
git clone https://github.com/khoi27122007-tech/SignFlow.git
cd SignFlow
```

### Bước 2. Tạo môi trường ảo

```bash
python -m venv .venv
```

Kích hoạt môi trường trên Windows:

```powershell
.venv\Scripts\activate
```

### Bước 3. Cài đặt thư viện

```bash
python -m pip install -r requirements.txt
```

Nếu muốn sử dụng GPU để tăng tốc quá trình huấn luyện, cần cài đặt phiên bản PyTorch tương thích với phần cứng và phiên bản CUDA của máy tính.

## 8. Hướng dẫn sử dụng

### Bước 1. Xây dựng dữ liệu Landmark

Chạy lệnh sau tại thư mục gốc của dự án:

```bash
python -m src.build_dataset --dataset "path/to/ASL_Dataset" --output "data/landmarks.npz"
```

Chương trình sẽ:

- Đọc hình ảnh trong từng thư mục nhãn.
- Phát hiện bàn tay bằng MediaPipe.
- Trích xuất 21 điểm mốc.
- Chuẩn hóa tọa độ.
- Chuyển đổi thành vector đặc trưng.
- Lưu dữ liệu vào file `landmarks.npz`.

### Bước 2. Huấn luyện mô hình

```bash
python -m src.train --data "data/landmarks.npz" --output "models/landmark_model.pth" --epochs 80
```

Quá trình huấn luyện bao gồm:

- Kiểm tra tính hợp lệ của dữ liệu.
- Chia dữ liệu thành tập huấn luyện và tập xác thực theo tỷ lệ phân tầng.
- Huấn luyện mô hình phân loại.
- Theo dõi Loss và Accuracy.
- Lưu checkpoint của mô hình tốt nhất.

### Bước 3. Khởi chạy nhận diện webcam

```bash
python -m src.webcam --model "models/landmark_model.pth" --threshold 0.65
```

Sau khi khởi chạy, hệ thống sẽ mở webcam và thực hiện nhận diện cử chỉ bàn tay theo thời gian thực.

Nhấn `Q` hoặc `ESC` để thoát chương trình.

### Các tham số tùy chọn

| Tham số | Chức năng |
|---|---|
| `--model` | Đường dẫn đến mô hình đã huấn luyện |
| `--camera` | Chỉ số thiết bị webcam |
| `--threshold` | Ngưỡng độ tin cậy dự đoán |
| `--smooth-window` | Số lượng dự đoán gần nhất được sử dụng để làm mượt |
| `--no-mirror` | Tắt chế độ lật gương hình ảnh webcam |

## 9. Mục đích sử dụng và ứng dụng thực tế

SignFlow được phát triển như một nền tảng thử nghiệm công nghệ nhận diện cử chỉ bàn tay, có tiềm năng mở rộng thành các ứng dụng hỗ trợ con người trong giao tiếp và tương tác với thiết bị số.

### 9.1. Hỗ trợ giao tiếp cho người khiếm thính và người gặp khó khăn trong giao tiếp bằng lời nói

Công nghệ nhận diện cử chỉ có thể giúp chuyển đổi các ký hiệu bàn tay đã nhận diện thành văn bản, góp phần hỗ trợ giao tiếp giữa người sử dụng ngôn ngữ ký hiệu và người chưa biết ngôn ngữ này.

### 9.2. Phát triển các ứng dụng hỗ trợ tiếp cận

Hệ thống có thể được mở rộng để tích hợp vào các phần mềm hỗ trợ người khuyết tật, tạo thêm phương thức tương tác với máy tính thông qua cử chỉ bàn tay.

### 9.3. Tương tác giữa con người và máy tính

Nhận diện cử chỉ bàn tay có thể trở thành một phương thức điều khiển thay thế cho bàn phím và chuột trong một số ứng dụng, đặc biệt là các hệ thống tương tác không tiếp xúc.

### 9.4. Hỗ trợ học tập ngôn ngữ ký hiệu

SignFlow có thể được phát triển thành công cụ hỗ trợ học tập, giúp người học luyện tập các tư thế bàn tay tương ứng với bảng chữ cái ASL và nhận phản hồi trực tiếp từ hệ thống.

### 9.5. Nghiên cứu và phát triển trí tuệ nhân tạo

Dự án cung cấp nền tảng thực hành và nghiên cứu các lĩnh vực:

- Thị giác máy tính (Computer Vision).
- Nhận diện tư thế bàn tay (Hand Pose Estimation).
- Phân loại dữ liệu bằng học sâu (Deep Learning Classification).
- Tương tác người – máy (Human-Computer Interaction).
- Công nghệ hỗ trợ (Assistive Technology).

**Lưu ý:** Phiên bản hiện tại nhận diện các tư thế bàn tay đơn lẻ thuộc bảng chữ cái ASL. Hệ thống chưa thực hiện dịch câu ngôn ngữ ký hiệu liên tục, chưa phân tích biểu cảm khuôn mặt và chưa xử lý ngữ pháp ngôn ngữ ký hiệu hoàn chỉnh.

## 10. Hạn chế hiện tại

Mặc dù có khả năng nhận diện theo thời gian thực, hệ thống vẫn còn một số hạn chế:

- Chất lượng nhận diện phụ thuộc vào khả năng phát hiện bàn tay của MediaPipe.
- Một số cử chỉ có hình dạng gần giống nhau có thể dẫn đến dự đoán nhầm.
- Điều kiện ánh sáng, góc quay và chất lượng webcam có thể ảnh hưởng đến kết quả.
- Phiên bản hiện tại chỉ xử lý một bàn tay được phát hiện.
- Mô hình phân loại các tư thế tĩnh độc lập, chưa nhận diện chuỗi cử chỉ động.
- Hiệu quả nhận diện phụ thuộc vào chất lượng, độ đa dạng và sự phân bố của dữ liệu huấn luyện.

## 11. Định hướng phát triển

Trong tương lai, SignFlow có thể được mở rộng theo các hướng:

- Mở rộng số lượng cử chỉ được nhận diện.
- Hỗ trợ nhận diện đồng thời hai bàn tay.
- Cải thiện độ ổn định trong các điều kiện ánh sáng và góc quay khác nhau.
- Xây dựng giao diện đồ họa trực quan (GUI).
- Phát triển chức năng ghép các chữ cái thành văn bản liên tục.
- Tích hợp chức năng chuyển văn bản thành giọng nói (Text-to-Speech).
- Nghiên cứu các mô hình học sâu có khả năng xử lý chuỗi thời gian để nhận diện cử chỉ động.
- Triển khai hệ thống thành ứng dụng máy tính hoặc nền tảng web.

## 12. Thông tin tác giả (Project Author)

| Thông tin | Nội dung |
|---|---|
| Tên dự án | SignFlow |
| Tác giả | Nguyễn Khôi |
| Lĩnh vực | Trí tuệ nhân tạo – Computer Vision |
| Công nghệ chính | Python, PyTorch, MediaPipe, OpenCV |
| Mục đích | Học tập, nghiên cứu và phát triển ứng dụng nhận diện cử chỉ bàn tay |
| GitHub | [khoi27122007-tech](https://github.com/khoi27122007-tech) |
| Repository | [SignFlow](https://github.com/khoi27122007-tech/SignFlow) |

Dự án được thực hiện với mục tiêu nghiên cứu, học tập và ứng dụng các kiến thức về trí tuệ nhân tạo, xử lý ảnh và học sâu vào bài toán nhận diện cử chỉ bàn tay.

## 13. Giấy phép và đóng góp

SignFlow được xây dựng phục vụ mục đích học tập, nghiên cứu và thử nghiệm công nghệ.

Mọi ý kiến đóng góp, đề xuất cải tiến hoặc phát hiện lỗi đều được hoan nghênh thông qua GitHub Issues và Pull Requests.

---

**SignFlow – Ứng dụng trí tuệ nhân tạo trong nhận diện cử chỉ bàn tay, hướng đến giao tiếp thuận tiện và dễ tiếp cận hơn.**
