from pathlib import Path
from datetime import datetime
import logging

__all__ = [
    "setup_logger",
]

def setup_logger(log_dir: str,
                 sub_path: str = "",
                 log_filename: str = "train_logs.log",
                 overwrite_log: bool = False,
                 log_to_console: bool = True):

    save_path = Path(log_dir) / sub_path
    save_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")
    name_stem = Path(log_filename).stem
    log_filepath = save_path / f"{name_stem}_{timestamp}.log"

    logger = logging.getLogger(str(log_filepath))
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    file_mode = 'w' if overwrite_log else 'a'

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)

    file_handler = logging.FileHandler(
        log_filepath, mode=file_mode, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    print(f"日志文件将保存到: {log_filepath}  （模式：{'覆盖' if overwrite_log else '追加'}）")
    print(f"控制台日志输出：{'开启' if log_to_console else '关闭'}")

    return logger
