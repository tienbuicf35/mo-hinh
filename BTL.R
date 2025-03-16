
# Tải các gói cần thiết
library(tidyverse)
library(sparklyr)
library(caret)
library(lattice)
library(zoo)
library(forecast)
library(gridExtra)

# Đảm bảo không có gói nào ghi đè không cần thiết
if ("package:quantmod" %in% search()) {
  detach("package:quantmod", unload = TRUE)
}

# Kết nối với Spark
spark_home <- "C:/Users/ACER/spark-3.5.5-bin-hadoop3/spark-3.5.5-bin-hadoop3"
sc <- spark_connect(master = "local", spark_home = spark_home)
on.exit(spark_disconnect(sc), add = TRUE)  # Đảm bảo ngắt kết nối khi thoát

# Kiểm tra kết nối
if (is.null(sc)) {
  stop("Kết nối Spark thất bại. Vui lòng kiểm tra đường dẫn spark_home, quyền truy cập, hoặc cài đặt Java.")
} else {
  print("Kết nối Spark thành công!")
}

# Đọc dữ liệu từ file CSV
data_path <- file.path("C:", "Users", "ACER", "Desktop", "archive (1)", "Global Health Statistics.csv")
data_spark <- spark_read_csv(
  sc,
  name = "thong_ke_suc_khoe_toan_cau",
  path = data_path,
  header = TRUE,
  infer_schema = TRUE

)

# Kiểm tra tên cột
colnames <- tbl_vars(data_spark)
print(colnames)

# Xem cấu trúc dữ liệu
sdf_describe(data_spark) %>% collect() %>% print()

# Kiểm tra giá trị thiếu
missing_counts <- data_spark %>%
  dplyr::summarise_all(~sum(is.na(.))) %>%
  collect()
print(missing_counts)

# Xử lý giá trị thiếu cho các cột số
numeric_cols <- c(
  "Tỷ lệ mắc bệnh (%)", "Tỷ lệ phát bệnh (%)", "Tỷ lệ tử vong (%)", "Tỷ lệ hồi phục (%)",
  "Bác sĩ trên 1000 người", "Giường bệnh trên 1000 người", "Dân số bị ảnh hưởng",
  "Thu nhập bình quân đầu người (USD)", "Chỉ số giáo dục", "Tỷ lệ đô thị hóa (%)",
  "Khả năng tiếp cận y tế (%)", "Chi phí điều trị trung bình (USD)", "DALYs",
  "Cải thiện trong 5 năm (%)"
)
data_spark <- impute_missing(data_spark, numeric_cols)

# Chuyển đổi kiểu dữ liệu
data_spark <- convert_data_types(data_spark)

# Bước 4: Phân tích khám phá (EDA)
# Xu hướng tỷ lệ mắc bệnh trung bình toàn cầu theo thời gian
global_trend <- data_spark %>%
  dplyr::group_by(Year) %>%
  dplyr::summarise(`Tỷ lệ mắc bệnh trung bình` = mean(`Tỷ lệ mắc bệnh (%)`, na.rm = TRUE)) %>%
  collect()
p1 <- create_plot(
  global_trend, x = "Year", y = "Tỷ lệ mắc bệnh trung bình", type = "line",
  title = "Xu hướng tỷ lệ mắc bệnh trung bình toàn cầu theo thời gian",
  x_lab = "Năm", y_lab = "Tỷ lệ mắc bệnh trung bình (%)"
)

# Tỷ lệ tử vong theo quốc gia (top 10)
mortality_by_country <- data_spark %>%
  dplyr::group_by(Country) %>%
  dplyr::summarise(`Tỷ lệ tử vong trung bình` = mean(`Tỷ lệ tử vong (%)`, na.rm = TRUE)) %>%
  collect() %>%
  dplyr::top_n(10, `Tỷ lệ tử vong trung bình`)
p2 <- create_plot(
  mortality_by_country, x = "Country", y = "Tỷ lệ tử vong trung bình", type = "bar",
  title = "Top 10 quốc gia có tỷ lệ tử vong cao nhất",
  x_lab = "Quốc gia", y_lab = "Tỷ lệ tử vong trung bình (%)", fill = "steelblue", coord_flip = TRUE
)

# Tỷ lệ hồi phục theo loại bệnh
recovery_by_disease <- data_spark %>%
  dplyr::group_by(`Loại bệnh`) %>%
  dplyr::summarise(`Tỷ lệ hồi phục trung bình` = mean(`Tỷ lệ hồi phục (%)`, na.rm = TRUE)) %>%
  collect()
p3 <- create_plot(
  recovery_by_disease, x = "Loại bệnh", y = "Tỷ lệ hồi phục trung bình", type = "bar",
  title = "Tỷ lệ hồi phục trung bình theo loại bệnh",
  x_lab = "Loại bệnh", y_lab = "Tỷ lệ hồi phục trung bình (%)", fill = "darkgreen", coord_flip = TRUE
)

# Bước 5: Dự đoán xu hướng tỷ lệ mắc bệnh (chuỗi thời gian)
disease_category <- data_spark %>%
  dplyr::select(`Loại bệnh`) %>%
  sparklyr::distinct() %>%
  collect() %>%
  dplyr::pull(`Loại bệnh`) %>%
  first()
disease_data <- data_spark %>%
  sparklyr::filter(`Loại bệnh` == disease_category) %>%
  collect()
ts_data <- ts(disease_data$`Tỷ lệ mắc bệnh (%)`, start = min(disease_data$Year), frequency = 1)
fit <- auto.arima(ts_data)
forecast_result <- forecast(fit, h = 5)
forecast_df <- as.data.frame(forecast_result)
forecast_df$Year <- seq(min(disease_data$Year), by = 1, length.out = nrow(forecast_df))
actual_df <- data.frame(Year = disease_data$Year, `Tỷ lệ mắc bệnh` = disease_data$`Tỷ lệ mắc bệnh (%)`)
p4 <- ggplot() +
  geom_line(data = actual_df, aes(x = Year, y = `Tỷ lệ mắc bệnh`), color = "black") +
  geom_line(data = forecast_df, aes(x = Year, y = `Point Forecast`), color = "blue", linetype = "dashed") +
  geom_ribbon(data = forecast_df, aes(x = Year, ymin = `Lo 95`, ymax = `Hi 95`), alpha = 0.2, fill = "blue") +
  labs(title = paste("Dự đoán xu hướng tỷ lệ mắc bệnh:", disease_category), x = "Năm", y = "Tỷ lệ mắc bệnh (%)") +
  theme_minimal()

# Bước 6: Hồi quy tuyến tính sử dụng Spark MLlib
feature_cols <- c(
  "Tỷ lệ mắc bệnh (%)", "Tỷ lệ phát bệnh (%)", "Tỷ lệ hồi phục (%)",
  "Bác sĩ trên 1000 người", "Giường bệnh trên 1000 người", "Dân số bị ảnh hưởng",
  "Thu nhập bình quân đầu người (USD)", "Chỉ số giáo dục", "Tỷ lệ đô thị hóa (%)",
  "Khả năng tiếp cận y tế (%)"
)

# Chia dữ liệu một lần duy nhất
set.seed(123)
splits <- sparklyr::sdf_random_split(data_spark, training = 0.8, test = 0.2)
train_data <- splits$training
test_data <- splits$test

# Hồi quy cho tỷ lệ tử vong
mortality_result <- train_and_evaluate(
  train_data, test_data, "Tỷ lệ tử vong (%)", feature_cols, "Tỷ lệ tử vong"
)
p5 <- mortality_result$plot
p7 <- create_plot(
  mortality_result$predictions, x = "Tỷ lệ tử vong (%)", y = "prediction", type = "scatter",
  title = "Dự đoán tỷ lệ tử vong theo quốc gia",
  x_lab = "Tỷ lệ tử vong thực tế (%)", y_lab = "Tỷ lệ tử vong dự đoán (%)", fill = "Country"
)

# Hồi quy cho tỷ lệ hồi phục
recovery_result <- train_and_evaluate(
  train_data, test_data, "Tỷ lệ hồi phục (%)", feature_cols, "Tỷ lệ hồi phục"
)
p6 <- recovery_result$plot
p8 <- create_plot(
  recovery_result$predictions, x = "Tỷ lệ hồi phục (%)", y = "prediction", type = "scatter",
  title = "Dự đoán tỷ lệ hồi phục theo quốc gia",
  x_lab = "Tỷ lệ hồi phục thực tế (%)", y_lab = "Tỷ lệ hồi phục dự đoán (%)", fill = "Country"
)

# Bước 7: Hiển thị tất cả biểu đồ
grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, ncol = 2)

# Bước 8: Tổng hợp kết quả
cat("\nKết quả mô hình:\n")
cat("RMSE cho dự đoán tỷ lệ tử vong:", mortality_result$rmse, "\n")
cat("RMSE cho dự đoán tỷ lệ hồi phục:", recovery_result$rmse, "\n")