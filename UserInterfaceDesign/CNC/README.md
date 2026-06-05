範例 Dash 應用程式

這個頁面目前是依照你的草圖做的第一版骨架，核心內容已經放進 `CNC/app.py`，樣式放在 `CNC/assets/style.css`。

執行方式：

```powershell
cd CNC
pip install dash dash-bootstrap-components pandas plotly
python app.py
```

打開 `http://127.0.0.1:8050`。

若要先用固定 6 段加工資料（建議展示時使用）：

```powershell
cd CNC
python generate_6stage_csv.py
python app.py
```

頁面內可用 `播放 CSV` 按鈕與 `CSV 播放索引` 進行播放/手動切換。

這版已經包含：

- 左側三個趨勢圖區塊
- 中間 STL / 工件示意區
- 右側狀態熱力圖、RA 盒狀圖、RA 趨勢圖
- 上方摘要卡片
- MQTT 即時接收 `cnc/snapshot`
- 返回按鈕與母畫面預覽送出

你新增的需求我先整理成需要確認的問題，避免我猜錯：

1. i 的參照點是固定工件序號，還是依時間軸/批次滑動？ 依照批次滑動的話，i-3 ... i+2 是相對於目前工件的前後的幾個工件
2. `i-3 ... i+2` 的資料來源是實際歷史資料、推論資料，還是先用模擬資料？ 先用 MQTT 假資料，等真實資料進來再替換
3. STL 模型檔案放在哪裡，要顯示哪一個工件？ `CNC\!Back Plate.stl`
4. 刀具磨耗、toque、bending、RA 的警戒線與錯誤線各是多少？ 刀具磨耗 0.25mm、toque 0.8Nm、bending max 80 / min 150、RA 1.5um
5. 狀態資料要怎麼傳出去對接其他系統：OPC UA、MQTT、HTTP API，還是 Dashboard？ 先用 API 或母畫面預覽，但不要 server POST 的匯出欄位
6. 右側分類顏色是否固定為斷刀風險、重刀、正常、輕刀四類？ 對，固定這四類
7. p1、p2、p3 對應的是哪些實際點位，還是只要支援多點位視覺化即可？ 目前先自動對照收到的數值，不要另外做多點位支援

補充：

- `CNC/fake_data_mqtt.py` 是獨立的 MQTT 假資料發送器。
- `CNC/app.py` 會訂閱 `cnc/snapshot`，所以先啟動 broker，再啟動 publisher，最後開 Dash。
- `CNC/parent_example.html` 是母畫面示範，會接收 child 的 postMessage。
- `CNC/CHANGELOG.md` 會記錄每次變更。
- 如果你要一次看 6 筆 snapshot，啟動發送器時加上 `--series 6`。

如果你要我直接往下一版做，我建議你回覆這 2 件事就夠了：

1. `返回` 要導回哪個母畫面路徑或 URL
2. `CNC/!Back Plate.stl` 是否已經放好，還是要我先做沒有 STL 的佔位版