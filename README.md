# DDM501 Tutorial 03 — Airflow và MLflow

Pipeline xử lý bộ dữ liệu WDBC theo 7 bước:

`ingest → validate → split → scale → train → register → report`

Airflow lập lịch và theo dõi từng bước. MLflow lưu chỉ số đánh giá, model artifact và phiên bản mô hình. Dự án chạy độc lập bằng Docker Compose; không cần stack của bài tutorial trước.

## Chạy bằng Docker

Yêu cầu: Docker Desktop đang chạy. Từ thư mục chứa `docker-compose.yml`:

```powershell
docker compose up -d --build
docker compose ps
docker compose exec airflow cat /opt/airflow/standalone_admin_password.txt
```

Đợi hai service báo `healthy`. Mở [Airflow](http://127.0.0.1:18080), đăng nhập bằng `admin` và mật khẩu vừa in. Mở [MLflow](http://127.0.0.1:15030) để xem experiment `wdbc-pipeline` và model `wdbc-classifier`.

Trên Linux, tạo file `.env` từ `.env.example` và thay `AIRFLOW_UID` bằng kết quả của `id -u` trước khi chạy Compose. Trên Windows, giá trị mặc định `50000` dùng được.

## Chạy DAG và xem kết quả

```powershell
docker compose exec airflow airflow dags test wdbc_pipeline 2026-08-25
docker compose exec airflow python scripts/fetch_and_predict.py --ds 2026-08-25
```

Trong `data/staging/2026-08-25/`, pipeline tạo bản chụp dữ liệu thô, dữ liệu hợp lệ, dòng bị loại, tập train/test, tham số scale, báo cáo kiểm tra và `summary.json`. File `data/staging/history.jsonl` giữ một dòng tóm tắt cho mỗi ngày logic. Chạy lại cùng ngày sẽ cập nhật đầu ra ngày đó và ghi đè dòng tóm tắt; mỗi lần chạy vẫn tạo một MLflow run và model version mới.

Các trường chính trong `summary.json`: số dòng hợp lệ, số dòng train/test, `accuracy`, `roc_auc`, `mlflow_run_id` và `model_version`. Bước scale chỉ tính trung bình và độ lệch chuẩn trên tập train để tránh rò rỉ dữ liệu test.

Script dự đoán tải mô hình từ MLflow Registry, không đọc mô hình trực tiếp từ thư mục Airflow. Có thể chọn phiên bản và số dòng:

```powershell
docker compose exec airflow python scripts/fetch_and_predict.py --version 1 --rows 10
```

Để chạy lại các ngày trước (backfill):

```powershell
docker compose exec airflow airflow dags backfill wdbc_pipeline -s 2026-08-22 -e 2026-08-24
```

Kiểm tra ba thư mục ngày tương ứng trong `data/staging/` và ba dòng ngày đó trong `history.jsonl`. Trên Airflow Grid, chọn từng run và từng task để xem log chi tiết. Mỗi ngày backfill cũng tạo một model version trong MLflow.

## Thử trường hợp dữ liệu lỗi

```powershell
python scripts/corrupt_extract.py
docker compose exec airflow airflow dags test wdbc_pipeline 2026-08-26
python scripts/corrupt_extract.py --repair
```

Script làm trống khoảng 12% giá trị `mean_radius`. Task `validate` ghi các dòng lỗi vào `rejected.parquet` rồi dừng DAG nếu hơn 5% dòng bị loại. Lệnh `--repair` khôi phục CSV gốc. Có thể xem lý do lỗi trong Airflow Grid → task `validate` → Logs.

## Cấu trúc chính

| Đường dẫn | Vai trò |
|---|---|
| `dags/wdbc_pipeline.py` | Định nghĩa DAG và 7 task |
| `scripts/fetch_and_predict.py` | Tải model từ Registry và dự đoán |
| `scripts/corrupt_extract.py` | Tạo và khôi phục dữ liệu lỗi để thử validation |
| `data/raw/wdbc.csv` | Dữ liệu đầu vào |
| `docker-compose.yml` | Service Airflow và MLflow |

`data/staging/`, `mlflow-data/`, `logs/` và `.env` là dữ liệu runtime, đã được bỏ qua trong Git.

## Kết quả kiểm thử

Với ngày logic `2026-08-25`, cả 7 task đều thành công: 564 dòng hợp lệ, 441 dòng train, 123 dòng test, `accuracy=0.9512`, `roc_auc=0.9956`. Model version 1 được đăng ký và script dự đoán tải lại version đó từ MLflow, dự đoán đúng 5/5 dòng thử. Backfill `2026-08-22` đến `2026-08-24` thành công cả 21 task, tạo ba thư mục ngày và model versions 2–4. Khi thêm dữ liệu lỗi, `validate` dừng DAG tại mức 13.0% dòng bị loại; lệnh `--repair` khôi phục CSV đúng checksum ban đầu.

Hướng triển khai MLflow tham khảo [bài mẫu của locnp13](https://github.com/locnp13/ddm501-t03-airflow).
