from __future__ import annotations

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

from motion_capture.config import load_config
from motion_capture.storage import AppDatabase
from motion_capture.ui.main_window import MainWindow


def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("MotionCapture")
    try:
        base_config = load_config()
        database = AppDatabase(base_config.database_path)
        config = database.load_config(base_config)
    except Exception as exc:
        QMessageBox.critical(None, "配置错误", str(exc))
        return 2
    window = MainWindow(config, database=database)
    window.show()
    return app.exec_()
