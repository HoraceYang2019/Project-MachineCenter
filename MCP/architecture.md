# MCP 工具說明書

## 目錄
1. [CSV_Getter](#csv-getter)
2. [main_MQTT_and_Reception](#main_mqtt_and_reception)
3. [render_path_on_stl_autoalign](#render_path_on_stl_autoalign)
4. [Siemens_mqtt](#siemens_mqtt)
5. [upload_to_NAS](#upload_to_nas)
6. [rpi_message_start](#rpi_message_start)
7. [rpi_message_end](#rpi_message_end)

---

## CSV_Getter

### 功能描述
從本地資料夾或遠端設備（透過SFTP）提取最新的CSV檔案，並將其複製到集中處理區。

### 主要函式

#### `get_latest_local_csv(src_dir, target_dir)`
- **用途**：從本地資料夾抓取最新的CSV檔案
- **參數**：
  - `src_dir`: 源資料夾路徑
  - `target_dir`: 目標資料夾路徑
- **回傳**：成功時回傳目標檔案路徑，失敗時回傳 `None`
- **特性**：
  - 使用檔案修改時間（mtime）判斷最新檔案

#### `get_latest_sftp_csv(host, port, username, password, remote_dir, target_dir)`
- **用途**：透過SFTP從遠端設備下載最新的CSV檔案
- **參數**：
  - `host`: 遠端設備IP位址
  - `port`: SSH連接埠（預設22）
  - `username`: SFTP使用者名稱
  - `password`: SFTP密碼
  - `remote_dir`: 遠端資料夾路徑
  - `target_dir`: 本地目標資料夾路徑
- **回傳**：成功時回傳下載檔案路徑，失敗時回傳 `None`
- **特性**：
  - 自動接受未知的SSH host key
  - 具有異常處理機制

#### `main(target_dir=None)`
- **用途**：主程式入口
- **參數**：
  - `target_dir`: 可選的目標資料夾路徑，若未指定則彈出UI選擇視窗
- **特性**：
  - 使用Tkinter UI讓使用者選擇目標資料夾
  - 預設位置：`C:\Users\user\Desktop\coding\mislab\SYNTEC\SYNTEC_CNC\logs`

### 依賴套件
- `paramiko`: SFTP連線
- `tkinter`: 檔案對話框UI
- `os`, `shutil`: 檔案操作

### 使用範例
```python
from CSV_Getter import get_latest_local_csv, get_latest_sftp_csv

# 從本地資料夾提取CSV
local_path = get_latest_local_csv("C:/source", "C:/target")

# 從遠端設備提取CSV
remote_path = get_latest_sftp_csv(
    "192.168.1.100", 22, "user", "password",
    "/remote/path", "C:/target"
)
```

---

## main_MQTT_and_Reception

### 功能描述
連接Syntec CNC機台，讀取即時資料（座標、轉速、進給率等），並透過MQTT發送到系統。支援無限重試連線機制。

### 主要類別

#### `SyntecCNCManager`
**用途**：管理與新代Syntec CNC的連線與資料讀取

**初始化參數**：無（自動從設定檔讀取）

**主要方法**：

- `connect()`
  - 用途：建立與CNC的連線，含無限重試機制
  - 特性：每5秒重試一次直到連線成功

- `read_data()`
  - 用途：讀取CNC的即時資料
  - 回傳：若成功回傳資料，發生異常時回傳 `None`
  - 讀取項目：
    - 座標位置 (X, Y, Z)
    - 機械座標
    - NC指針位置
    - 轉速（每4次讀取更新一次）
    - 進給率（每4次讀取更新一次）
    - 工件及機械座標系
  - 特性：
    - 約1200~1600ms更新一次轉速和進給率
    - 具有延遲計算功能
    - 會快取初始EXT和G54座標

#### `AsyncMQTTManager`
**用途**：管理異步MQTT連線與訊息發布

**初始化參數**：
- `broker`: MQTT代理器位址（預設："10.0.0.103"）
- `topic`: MQTT主題（預設："STH/CNC"）

**主要方法**：
- `connect()`: 連接到MQTT代理器，含重試機制
- `on_connect()`, `on_message()`, `on_publish()`: MQTT回調函式
- `loop_start()`, `loop_stop()`: 控制MQTT事件循環

### 依賴套件
- `paho-mqtt`: MQTT通訊
- `numpy`: 數值計算
- `paramiko`: SFTP連線
- `config`: 專案設定模組（需自行定義）

### 使用流程
```python
from main_MQTT_and_Reception import SyntecCNCManager, AsyncMQTTManager

# 初始化CNC管理器
cnc_manager = SyntecCNCManager()
cnc_manager.connect()

# 初始化MQTT管理器
mqtt_manager = AsyncMQTTManager()
mqtt_manager.connect()
mqtt_manager.loop_start()

# 主讀取循環
while True:
    data = cnc_manager.read_data()
    if data:
        mqtt_manager.publish_data(data)
```

### 設定需求
- 需要 `config` 模組，包含：
  - `readconfig()`: 讀取IP、端口等設定
  - `ref_syntecAPi()`: 初始化Syntec API連線

---

## render_path_on_stl_autoalign

### 功能描述
讀取CNC加工路徑CSV檔案，將其3D路徑渲染在STL模型上，並根據參考資料自動對齊。生成互動式HTML視覺化。

### 主要函式

#### `load_and_merge(data_csv, ref_csv)`
- **用途**：合併加工資料和參考資料
- **參數**：
  - `data_csv`: 加工資料CSV檔案路徑
  - `ref_csv`: 參考資料CSV檔案路徑
- **回傳**：合併後的Pandas DataFrame
- **必需欄位**（data_csv）：
  - `Offset_X`, `Offset_Y`, `Offset_Z`
  - `Torque`, `BendingX`
- **必需欄位**（ref_csv）：
  - `Offset_X`, `Offset_Y`, `Offset_Z`
  - `Torque_UCL_3`, `Torque_UCL_5`
  - `BendingX_UCL_3`, `BendingX_UCL_5`

#### `stl_to_mesh3d(stl_path)`
- **用途**：將STL檔案轉換為可視化的mesh數據
- **參數**：`stl_path`: STL檔案路徑
- **回傳**：`(unique_vertices, faces)` 元組

#### `recenter_stl_to_xy_center_top_z(verts)`
- **用途**：重新定位STL模型
- **功能**：
  - XY原點移至模型中心
  - Z=0設置為模型頂部
- **參數**：`verts`: 頂點陣列（Nx3）
- **回傳**：已移位的頂點陣列

#### `_bbox_xyz(points)`
- **用途**：計算點集的邊界框
- **回傳**：`(minx, maxx, miny, maxy, minz, maxz, cx, cy)` 元組

### 預設路徑設定
```python
DEFAULT_DATA_CSV = "output/aligned_no_ucl.csv"
DEFAULT_REF_CSV = "output/reference.csv"
DEFAULT_STL = "data/325BTM.STL"
OUTPUT_DIR = "output"
```

### 色階設定
路徑根據數值以色階顯示：
- 藍色 (0.0): 低值
- 橙色 (0.5): 中值
- 紅色 (1.0): 高值

### 命令列參數
```bash
python render_path_on_stl_autoalign.py \
  --data-csv <path> \
  --ref-csv <path> \
  --stl-file <path> \
  --align-method [auto-align-top|recenter]
```

### 依賴套件
- `numpy`: 數值計算
- `pandas`: 資料處理
- `plotly`: 互動式視覺化
- `numpy-stl`: STL檔案處理

### 輸出
- HTML檔案：`output/path_on_stl.html`（互動式可視化）

---

## Siemens_mqtt

### 功能描述
連接PLC（使用SNAP7協議）讀取資料，並透過MQTT發送到系統。具備自動重連機制。

### 主要類別

#### `CNCManager`
**用途**：管理與PLC的S7連線並讀取資料

**初始化參數**：
- `ip`: PLC IP位址（預設："192.168.3.10"）
- `rack`: PLC機架號（預設：0）
- `slot`: PLC插槽號（預設：2）
- `port`: 連接埠（預設：102）

**主要方法**：

- `connect()`
  - 用途：連接PLC，含無限重試機制
  - 特性：每5秒重試一次

- `read_data()`
  - 用途：讀取PLC資料區（DB710）
  - 回傳：包含6個浮點數的陣列 `[工件X, 工件Y, 工件Z, 機械X, 機械Y, 機械Z]`
  - 特性：具有自動重連機制
  - 讀取欄位：
    - 位元組 0-3: 工件X座標
    - 位元組 4-7: 工件Y座標
    - 位元組 8-11: 工件Z座標
    - 位元組 20-23: 機械X座標
    - 位元組 24-27: 機械Y座標
    - 位元組 28-31: 機械Z座標

- `disconnect()`
  - 用途：安全關閉PLC連線

#### `AsyncMQTTManager`
**用途**：管理MQTT連線與訊息發布

**初始化參數**：
- `broker`: MQTT代理器位址（預設："10.0.0.103"）
- `topic`: MQTT主題（預設："STH/CNC"）

**主要方法**：
- `connect()`: 連接MQTT代理器
- `publish_message()`: 發布訊息
- `on_connect()`, `on_message()`: 回調函式

### 依賴套件
- `snap7`: S7協議PLC通訊
- `paho-mqtt`: MQTT通訊
- 標準庫：`threading`, `queue`, `json`, `datetime`

### 通訊協議
- **PLC連線**：S7通訊（STEP 7）
- **MQTT主題**：`STH/CNC`
- **訊息格式**：JSON

### 使用流程
```python
from Siemens_mqtt import CNCManager, AsyncMQTTManager

# 初始化PLC管理器
plc_manager = CNCManager(ip="10.0.0.69")
plc_manager.connect()

# 初始化MQTT管理器
mqtt_manager = AsyncMQTTManager()
mqtt_manager.connect()

# 讀取並發送資料
while True:
    data = plc_manager.read_data()
    if data:
        mqtt_manager.publish_message("STH/CNC", {"coordinates": data})
    time.sleep(0.5)
```

---

## upload_to_NAS

### 功能描述
將本地資料夾的所有內容透過SFTP上傳到NAS伺服器，支援遞歸建立遠端資料夾。

### NAS設定（预置）
```python
nas_ip = "163.18.48.141"
username = "admin"
password = "Nkust@0000"
remote_base_directory = "/share/FilePool/MachineCenter"
```

### 主要函式

#### `sftp_mkdir_p(sftp, remote_path)`
- **用途**：在SFTP伺服器上遞歸建立資料夾（類似 `mkdir -p`）
- **參數**：
  - `sftp`: SFTP連線物件
  - `remote_path`: 遠端資料夾路徑
- **回傳**：成功時回傳 `True`，失敗時回傳 `False`
- **特性**：
  - 自動處理路徑分隔符
  - 逐層建立不存在的資料夾

#### `upload_directory_to_nas(local_path, remote_target_path)`
- **用途**：遞歸上傳本地資料夾到NAS
- **參數**：
  - `local_path`: 本地資料夾路徑
  - `remote_target_path`: NAS上的目標路徑（完整路徑）
- **流程**：
  1. 連接到NAS
  2. 建立遠端目標資料夾
  3. 遍歷本地資料夾結構
  4. 在遠端創建對應的子資料夾
  5. 上傳所有檔案
- **特性**：
  - 會自動建立本地資料夾對應的遠端資料夾結構
  - 具有詳細的進度報告

### 依賴套件
- `paramiko`: SFTP連線
- `tkinter`: 檔案對話窗（可選）
- 標準庫：`os`

### 使用範例
```python
from upload_to_NAS import upload_directory_to_nas

# 上傳資料夾到NAS
local_folder = "C:/path/to/local/folder"
remote_path = "/share/FilePool/MachineCenter/202509089_TT"

upload_directory_to_nas(local_folder, remote_path)
```

### 路徑處理
- **本地路徑**：使用系統分隔符
- **遠端路徑**：必須使用正斜線 `/`
- 自動轉換本地路徑分隔符為SFTP標準格式

### 錯誤處理
- 連線失敗時會輸出錯誤訊息並中止
- 資料夾建立失敗時會跳過該目錄
- 單個檔案上傳失敗時會輸出錯誤但繼續上傳其他檔案

---

## rpi_message_start

### 功能描述
透過MQTT發送系統啟動配置訊息到樹莓派（或其他數據接收端）。初始化系統配置並啟動加工參考模式。

### 初始化流程
1. 初始化配置管理器
2. 初始化數據管理器
3. 初始化工作處理器
4. 建立MQTT處理器
5. 等待MQTT連線成功

### 主要函式

#### `wait_for_mqtt(mqtt_handler, timeout=5)`
- **用途**：等待MQTT連成功
- **參數**：
  - `mqtt_handler`: MQTT處理器物件
  - `timeout`: 等待超時時間（秒）
- **回傳**：連線成功回傳 `True`，超時回傳 `False`

#### `main(mac)`
- **用途**：主程式入口
- **參數**：`mac`: 設備MAC位址
- **發送訊息**：

**訊息1 - 配置訊息**
```json
{
    "CLASS": "CONFIG",
    "TYPE": "SetConfig",
    "VER": "01",
    "ID": 556,
    "PYD": {
        "SysTime": "YYYY-MM-DD HH:MM:SS",
        "TMAC": "MAC地址"
    }
}
```

**訊息2 - 啟動訊息**
```json
{
    "CLASS": "START",
    "TYPE": "Start",
    "VER": "01",
    "ID": 5678,
    "PYD": {
        "Mode": "Reference",
        "IsDetail": true
    }
}
```

### 時序說明
- 發送CONFIG訊息
- 等待0.5秒（樹莓派最低MQTT處理延遲）
- 發送START訊息

### 執行方式
```python
if __name__ == "__main__":
    main("6055f9e3d0d6")  # 傳入MAC位址
```

### 依賴模組
- `config`: 配置管理
- `data_manager`: 數據管理
- `mqtt_handler`: MQTT通訊
- `job_handler`: 工作處理
- 標準庫：`time`, `logging`

### 日誌記錄
- 使用 Python `logging` 模組
- 記錄MQTT連線狀態和訊息發送狀況

---

## rpi_message_end

### 功能描述
透過MQTT發送系統停止訊息到樹莓派（或其他數據接收端）。結束加工任務並保存加工模式。

### 初始化流程
與 `rpi_message_start.py` 相同：
1. 初始化配置管理器
2. 初始化數據管理器
3. 初始化工作處理器
4. 建立MQTT處理器
5. 等待MQTT連線成功

### 主要函式

#### `wait_for_mqtt(mqtt_handler, timeout=5)`
- 同 `rpi_message_start.py`

#### `main(mac)`
- **用途**：主程式入口
- **參數**：`mac`: 設備MAC位址
- **發送訊息**：

**停止訊息**
```json
{
    "CLASS": "RUN",
    "TYPE": "Stop",
    "VER": "01",
    "ID": 12345,
    "PYD": {
        "SavePattern": true
    }
}
```

### 訊息特性
- `SavePattern: true` - 表示保存本次加工樣式
- 發送後等待0.2秒確保訊息被處理

### 執行方式
```python
if __name__ == "__main__":
    main()  # 傳入MAC位址
```

### 依賴模組
- 同 `rpi_message_start.py`

### 用途場景
- 加工任務完成時呼叫
- 系統正常或異常中止時呼叫
- 保存當次加工的參考模式和參數

---

## 裝置連線互連圖

```
                                                  ┌─────────────────────┐ 
                                                  │         STH         │
                                                  │  rpi_message_end.py |
                                                  │ rpi_message_start.py|
                                                  └─────────────────────┘
                                                            ▲                      
                                                            │ 
   ┌────────┴─────────┐     ┌──────┴──────┐       ┌─────────────────────┐ 
   │   CNC SYNTEC│    │     │ CNC Siemens │       │         樹梅派       │
   │ (main_MQTT...)   │ OR  │(Siemens_mqtt)│      │  rpi_message_end.py |
   └────────┬─────────┘     └──────┬──────┘       │ rpi_message_start.py| 
            │                      │              └─────────────────────┘
            └──────────┬───────────┘                     ▲
                       │                                 │ 
                       ▼                                 ▼
              ┌─────────────────────────────────────────────┐
              |                     router                  |
              │                 (10.0.0.103)                │
              └─────────────────────────────────────────────┘
                       ▲ 
                       │
                       │ 
              ┌────────▼─────────┐
              │     TT server    │
              │   (預計工業電腦)  │
              └─────────┬────────┘
                        │     (外網)
              ┌─────────▼────────┐
              │   NAS Storage    │
              │(upload_to_NAS.py)│
              └──────────────────┘
```

---

## 資料流程圖

```
CSV Data Collection
         ↓
    CSV_Getter.py
   (Local/SFTP)
         ↓
   Central Directory
         edge 
         ↓
render_path_on_stl.py
   (Visualization)
         ↓
   Interactive HTML
   
Parallel:
CNC/PLC Data Stream
    ↓              ↓
Syntec API (新代) SNAP7 API (西門子)
    ↓              ↓
main_MQTT.. Siemens_mqtt
    ↓              ↓
    └──→ MQTT Broker ←─┐
         ↓             │
    rpi_message_start  │
    rpi_message_end    │
         │             │
         └─→ NAS Upload←┘
```


## 常用MQTT訊息格式

所有MQTT訊息遵循以下標準JSON格式：

```json
{
    "CLASS": "訊息分類 (CONFIG/START/RUN/DATA)",
    "TYPE": "訊息類型 (SetConfig/Start/Stop/etc)",
    "VER": "01",
    "ID": 訊息ID號碼,
    "PYD": {
        "訊息內容": "值"
    }
}
```

---

## 環境變量與設定

### Python環境
- Python 3.8+
- 建議使用虛擬環境

### 必需套件
```
paho-mqtt>=1.7.0
paramiko>=2.11.0
numpy>=1.21.0
pandas>=1.3.0
plotly>=5.0.0
numpy-stl>=3.0.0
snap7>=1.4.5
```

### 系統配置檔
各工具需要的配置應寫入 `config.py` 或相應的配置檔案：
- CNC IP位址
- PLC IP位址
- MQTT Broker位址
- NAS認證資訊
- 工作資料夾路徑

---

## 故障排除
- 新代CNC連線問題
  - 確認CNC IP位址和連接埠 (10.0.0.69)
-STH MQTT連線問題
  - 確認MQTT Broker IP位址 (10.0.0.103)
### 連線問題
- 檢查是否為同一局域網
- 確認防火牆設定
- 驗證IP位址和連接埠

### MQTT連線失敗
- 確認MQTT Broker正在運行
- 檢查代理器IP和連接埠
- 檢查網路連線

### 檔案操作失敗
- 檢查本地資料夾權限
- 檢查NAS認證資訊
- 確認遠端路徑存在或有建立權限

### 資料讀取異常
- 檢查CNC/PLC連線狀態
- 查看日誌檔案
- 驗證DB編號和位元組位移

---

**文檔版本**：v1.0  
**最後更新**：2026-08-12  
