# gui/components/training_monitor.py

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import numpy as np

class TrainingMonitor:
    def __init__(self, canvas_key, progress_key, status_keys):
        """
        初始化训练监控组件
        :param canvas_key: GUI画布元素的键
        :param progress_key: 进度条元素的键
        :param status_keys: 状态元组（当前epoch键, 最佳loss键）
        """
        # 初始化绘图数据
        self.train_loss = []
        self.val_loss = []
        self.metrics = {}
        
        # 创建Matplotlib图形
        self.fig, self.ax = plt.subplots(figsize=(14, 10))  # 单位为英寸，dpi默认100
        self.setup_plot()
        
        # GUI元素键
        self.canvas_key = canvas_key
        self.progress_key = progress_key
        self.current_epoch_key, self.best_loss_key = status_keys
        
        # 画布引用（将在绑定窗口后设置）
        self.canvas = None

    def setup_plot(self):
        """配置绘图样式"""
        self.ax.set_title("Training Progress")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True)
        self.train_line, = self.ax.plot([], [], label='Train Loss', color='blue')
        self.val_line, = self.ax.plot([], [], label='Val Loss', color='orange')
        self.ax.legend()

    def attach_to_window(self, window):
        """将绘图绑定到指定窗口"""
        tk_canvas = window[self.canvas_key].TKCanvas
        self.canvas = FigureCanvasTkAgg(self.fig, tk_canvas)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        self.sg_window = window  # 保存 PySimpleGUI 的 window

    def update_metrics(self, train_loss, val_loss, current_epoch, total_epochs, best_loss=None):
        """
        更新监控指标
        :param train_loss: 当前训练损失
        :param val_loss: 当前验证损失
        :param current_epoch: 当前epoch数
        :param total_epochs: 总epoch数
        :param best_loss: 最佳验证损失
        """
        # 更新数据
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        
        # 更新曲线
        x_data = np.arange(1, len(self.train_loss)+1)
        self.train_line.set_data(x_data, self.train_loss)
        self.val_line.set_data(x_data, self.val_loss)
        
        # 调整坐标轴范围
        self.ax.relim()
        self.ax.autoscale_view()
        
        # 更新GUI元素
        if self.canvas:
            self.canvas.draw()
        
        # 更新进度和状态
        self.update_progress(current_epoch, total_epochs)
        self.update_status(current_epoch, best_loss)

    def update_progress(self, current, total):
        """更新进度条"""
        progress = int((current / total) * 100) if total != 0 else 0
        if hasattr(self, 'sg_window') and self.sg_window is not None:
            self.sg_window[self.progress_key].update(progress)

    def update_status(self, current_epoch, best_loss):
        """更新状态文本"""
        if hasattr(self, 'sg_window') and self.sg_window is not None:
            self.sg_window[self.current_epoch_key].update(current_epoch)
            if best_loss is not None:
                self.sg_window[self.best_loss_key].update(f"{best_loss:.4f}")

    def reset(self):
        """重置监控器状态"""
        self.train_loss.clear()
        self.val_loss.clear()
        self.train_line.set_data([], [])
        self.val_line.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()
        if self.canvas:
            self.canvas.draw()
        self.update_progress(0, 1)
        self.update_status(0, None)

    def add_metric(self, name, values):
        """添加自定义指标（需手动处理绘制）"""
        self.metrics[name] = values

# 使用示例
if __name__ == "__main__":
    # 创建测试窗口
    layout = [
        [sg.Canvas(key='-CANVAS-')],
        [sg.ProgressBar(100, key='-PROGRESS-')],
        [sg.Text("当前轮次: 0", key='-EPOCH-'), sg.Text("最佳损失: --", key='-BEST-')],
        [sg.Button('模拟训练')]
    ]
    window = sg.Window("Training Monitor Test", layout, finalize=True)

    # 初始化监控器
    monitor = TrainingMonitor(
        canvas_key='-CANVAS-',
        progress_key='-PROGRESS-',
        status_keys=('-EPOCH-', '-BEST-')
    )
    monitor.attach_to_window(window)

    # 模拟训练循环
    epoch = 0
    while True:
        event, _ = window.read(timeout=100)
        if event == sg.WIN_CLOSED:
            break
        if event == '模拟训练':
            if epoch >= 50:
                epoch = 0
                monitor.reset()
            train_loss = np.random.rand() * (50 - epoch)/50
            val_loss = train_loss + np.random.rand()*0.1
            monitor.update_metrics(
                train_loss=train_loss,
                val_loss=val_loss,
                current_epoch=epoch,
                total_epochs=50,
                best_loss=min(monitor.val_loss) if monitor.val_loss else None
            )
            epoch += 1

    window.close()