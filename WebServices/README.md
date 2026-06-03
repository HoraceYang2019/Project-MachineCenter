# CNC 智慧製造系統 - Web Services 總索引

## 概述

本目錄包含 CNC 智慧製造系統的所有預測和診斷 Web 服務（API）。  
所有服務均基於 FastAPI 構建，支援 JSON 輸入輸出，可透過 MQTT、OPC-UA 聯動。

---

## 已支援服務一覽

| 服務名稱 | 端點 | 功能 | 文件 |
|---------|------|------|------|
| 表面粗糙度預測 | `/api/predict/ra` | 根據加工特徵預測工件表面粗糙度 (Ra) | [詳見](./ra-prediction.md) |
| 刀具磨損預測 | `/api/predict/tool-wear` | 預測刀具磨損量 (VB)，評估刀具壽命 | [待實現] |
| 品質診斷 | `/api/diagnose/quality` | 即時診斷加工品質問題 | [待實現] |
| 刀具壽命評估 | `/api/assess/tool-life` | 綜合預測，建議何時更換刀具 | [待實現] |

---

## 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 環境設定

複製 `.env.example` 為 `.env`：

```bash
MODEL_PATH=./models
MODEL_VERSION=rf-ra-v1.2
DATABASE_URL=postgresql://user:pass@localhost/cnc_db
DEBUG=false
```

### 3. 啟動服務

```bash
# FastAPI + Uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000

# 搭配 Celery（排程任務）
celery -A tasks worker --loglevel=info
```

### 4. 驗證服務

訪問 Swagger UI：
- **Swagger**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## API 通用規範

### Request 格式

```bash
POST /api/{category}/{service_name}
Content-Type: application/json
Authorization: Bearer {token}  # 若啟用認證

{
  "window_id": "w10233",
  "machine_id": "TMV-720",
  "feature_1": value1,
  "feature_2": value2
}
```

### Response 格式 (Success: 200)

```json
{
  "status": "success",
  "data": {
    "prediction_id": "Pred_xxx",
    "target": "service_name",
    "value": 1.28,
    "unit": "um",
    "confidence": 0.91,
    "model_version": "rf-ra-v1.2",
    "timestamp": "2026-04-13T10:00:00.400Z"
  }
}
```

### Error 格式 (4xx, 5xx)

```json
{
  "status": "error",
  "message": "Descriptive error message",
  "code": "ERROR_CODE",
  "details": {}
}
```

---

## 通用錯誤碼

| 狀態碼 | 錯誤碼 | 說明 |
|-------|--------|------|
| 200 | - | 成功 |
| 400 | `VALIDATION_ERROR` | 輸入格式錯誤 |
| 422 | `VALUE_OUT_OF_RANGE` | 輸入值超出允許範圍 |
| 401 | `UNAUTHORIZED` | 認證失敗 |
| 404 | `NOT_FOUND` | 資源不存在 |
| 500 | `INTERNAL_ERROR` | 伺服器錯誤 |
| 503 | `MODEL_UNAVAILABLE` | 模型服務不可用 |

---

## 資料庫結構

### 預測結果表 (predictions)

```sql
CREATE TABLE predictions (
  prediction_id VARCHAR PRIMARY KEY,
  window_id VARCHAR,
  machine_id VARCHAR,
  target VARCHAR,
  value FLOAT,
  unit VARCHAR,
  confidence FLOAT,
  model_version VARCHAR,
  timestamp TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_window ON predictions(window_id);
CREATE INDEX idx_predictions_machine ON predictions(machine_id);
CREATE INDEX idx_predictions_timestamp ON predictions(created_at);
```

### 信號特徵表 (signal_features)

```sql
CREATE TABLE signal_features (
  feature_id VARCHAR PRIMARY KEY,
  window_id VARCHAR,
  machine_id VARCHAR,
  torque_std FLOAT,
  torque_mean FLOAT,
  torque_max FLOAT,
  spindle_speed_rpm FLOAT,
  feed_rate_mm_min FLOAT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_signal_features_window ON signal_features(window_id);
```

---

## 排程與監控

### 日常排程

| 任務 | 時間 | 說明 |
|------|------|------|
| `batch_predict_daily` | 隔日 01:00 | 批量預測昨天所有加工資料 |
| `quality_summary_hourly` | 每小時 :00 | 產生品質小時統計 |
| `model_health_check` | 每 6 小時 | 驗證模型可用性 |

### Celery 任務示例

```bash
# 查看工作佇列
celery -A tasks inspect active

# 查看已完成任務
celery -A tasks inspect reserved

# 清空佇列
celery -A tasks purge
```

---

## 文件結構

```
WebServices/
├── README.md                 ← 你在這裡（總索引）
├── ra-prediction.md          ← 表面粗糙度預測服務
├── tool-wear-prediction.md   ← 刀具磨損預測服務（規劃中）
├── app.py                    ← FastAPI 主程式
├── tasks.py                  ← Celery 排程任務
├── models/                   ← 預測模型檔案
│   ├── rf-ra-v1.2.pkl
│   └── gb-toolwear-v1.0.pkl
├── requirements.txt
├── .env.example
└── tests/
    ├── test_ra_prediction.py
    └── test_api.py
```

---

## 開發指南

### 新增服務步驟

1. **建立服務文件**
   ```bash
   touch my-service.md
   ```

2. **在 README.md 服務表格中新增一行**

3. **在 app.py 中實現端點**
   ```python
   @app.post("/api/predict/my-service")
   async def predict_my_service(request: MyServiceRequest):
       # 實現預測邏輯
       pass
   ```

4. **新增單元測試**
   ```bash
   pytest tests/test_my_service.py
   ```

5. **更新 Swagger 文件**
   服務定義好後自動生成

### 測試 API

```bash
# 使用 curl
curl -X POST http://localhost:8000/api/predict/ra \
  -H "Content-Type: application/json" \
  -d '{"window_id": "w10233", "machine_id": "TMV-720", "torque_std": 0.079, ...}'

# 使用 Python requests
import requests
response = requests.post(
    'http://localhost:8000/api/predict/ra',
    json={"window_id": "w10233", ...}
)
print(response.json())
```

---

## 注意事項

1. **模型版本管理**: 每次更新模型後務必更新版本號，儲存預測時必須記錄版本
2. **輸入驗證**: 所有輸入需驗證類型和範圍，不符合返回 422
3. **可追溯性**: 每筆預測都要記錄 `model_version`、`confidence`、`timestamp`
4. **效能**: 單一預測 < 50ms；批次預測依量調整（建議單次 ≤ 1000 條）
5. **失敗重試**: 建議使用 Celery retry 機制處理暫時故障

---

## 聯繫與支援

- **模型相關**: MISLAB (Intelligent Sensing Lab)
- **API 聯動**: 參考 [CNC 主系統文件](../CNC/)
- **資料來源**: [MachineCenter 範例資料](../MahineCenter_0/examples/)

---

**最後更新**: 2026-06-02  
**維護者**: MISLAB  
**License**: NKUST Internal Use Only
