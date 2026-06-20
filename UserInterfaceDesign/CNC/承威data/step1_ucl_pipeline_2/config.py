# 配置管理模組，負責載入與管理 config.json 檔案
import json
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_path="./config.json"):
        # 初始化配置檔案路徑
        self.CONFIG_PATH = config_path
        # 補上這行：設定要計算 3 倍與 5 倍的標準差
        self.UCL_SIGMA = [3, 5]
        # 載入配置檔案
        self.load_config()

    def load_config(self, return_formatted=False):
        """
        載入 config.json 檔案並更新全域配置變數
        若 return_formatted=True，則返回格式化的 JSON 字串
        """
        try:
            with open(self.CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
            # 更新配置屬性
            self.MQTT_BROKER = config["MQTT_BROKER"]
            self.MQTT_PORT = config["MQTT_PORT"]
            self.PDATA_TOPIC = config["PDATA_TOPIC"]
            self.CYCLE_TOPIC = config["CYCLE_TOPIC"]
            self.CNC_TOPIC = config["CNC_TOPIC"]
            self.CNC_IP_TOPIC = config["CNC_IP_TOPIC"]
            self.CNC_BUTTON_TOPIC = config["CNC_BUTTON_TOPIC"]
            self.CNC_RE_TOPIC = config["CNC_RE_TOPIC"]
            self.CMD_REQ_TOPIC = config["CMD_REQ_TOPIC"]
            self.CMD_RESP_TOPIC = config["CMD_RESP_TOPIC"]
            self.MAC_TOPIC = config["MAC_TOPIC"]
            self.CNC_IP = config["CNC_IP"]
            self.YAML_DIR = config["YAML_DIR"]
            self.CSV_DIR = config["CSV_DIR"]
            self.COEFF_PATH = config["COEFF_PATH"]
            self.CSS_PATH = config["CSS_PATH"]
            self.PORT = config["PORT"]
            self.PDATA_LENGTH = config["PDATA_LENGTH"]
            self.CNC_LENGTH = config["CNC_LENGTH"]
            self.FLUTE_VIEW_LENGTH = config["FLUTE_VIEW_LENGTH"]
            self.WEB_UPDATE_RATE = config["WEB_UPDATE_RATE"]
            self.AUTO_REPORT_ON_CYCLE_END = config["AUTO_REPORT_ON_CYCLE_END"]
            self.TORQUE_UCL = config["TORQUE_UCL"]
            self.BENDING_UCL = config["BENDING_UCL"]
            self.FLUTE_UCL = config["FLUTE_UCL"]
            self.WEB_VERSION = config["WEB_VERSION"]
            self.HOLDE_MAIN_VERSION = config["HOLDE_MAIN_VERSION"]
            if return_formatted:
                return json.dumps(config, indent=2)
            return config
        except Exception as e:
            logger.error(f"載入 config.json 失敗: {e}")
            if return_formatted:
                return "Error loading config.json"
            return {}

    def save_config(self, config_content):
        """
        儲存編輯後的 config.json 內容
        """
        try:
            with open(self.CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(config_content)
            self.load_config()  # 更新配置
            # logger.info("config.json 儲存並更新成功")
            return True, ""
        except json.JSONDecodeError as e:
            logger.error(f"無效的 JSON 內容: {e}")
            return False, f"Invalid JSON content: {str(e)}"
        except Exception as e:
            logger.error(f"儲存 config.json 失敗: {e}")
            return False, f"Error saving config.json: {str(e)}"

    def load_coeff(self, return_formatted=False):
        """
        載入 coefficient.csv 檔案並返回內容
        若 return_formatted=True，則返回格式化的 CSV 字串
        """
        try:
            with open(self.COEFF_PATH, "r", encoding="utf-8") as f:
                content = f.read()
            if return_formatted:
                return content
            return content
        except Exception as e:
            logger.error(f"載入 coefficient.csv 失敗: {e}")
            return "" if return_formatted else ""

    def save_coeff(self, coeff_content):
        """
        儲存編輯後的 coefficient.csv 內容
        """
        try:
            with open(self.COEFF_PATH, "w", encoding="utf-8") as f:
                f.write(coeff_content)
            logger.info("coefficient.csv 儲存成功")
            return True, ""
        except Exception as e:
            logger.error(f"儲存 coefficient.csv 失敗: {e}")
            return False, f"Error saving coefficient.csv: {str(e)}"