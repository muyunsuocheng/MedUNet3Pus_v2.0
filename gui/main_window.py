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
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
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
        self.sg_window = window  # 保存 PySimpleGUI window

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

class MainWindow:
    def __init__(self, canvas_key, progress_key, status_keys):
        # 参数区
        param_layout = [
            [sg.Text("网络参数配置", font=('微软雅黑', 18))],
            [sg.Text('训练数据', font=('微软雅黑', 14)), sg.Input(key='-TRAIN_DATA-', font=('微软雅黑', 14)), sg.FolderBrowse('选择训练数据', font=('微软雅黑', 14))],
            [sg.Text('Labels', font=('微软雅黑', 14)), sg.Input(key='-LABELS-', font=('微软雅黑', 14)), sg.FolderBrowse('选择Labels', font=('微软雅黑', 14))],
            [sg.Text('输入通道', font=('微软雅黑', 14)), sg.Input('1', key='-IN_CH-', size=8, font=('微软雅黑', 14))],
            [sg.Text('类别数', font=('微软雅黑', 14)), sg.Input('3', key='-NUM_CLASS-', size=8, font=('微软雅黑', 14))],
            [sg.Text('特征缩放', font=('微软雅黑', 14)), sg.Slider((1,8), 4, 1, key='-FEATURE_SCALE-', font=('微软雅黑', 14), orientation='h', size=(20,20))],
            [sg.Text('迭代次数', font=('微软雅黑', 14)), sg.Input('10', key='-EPOCHS-', size=8, font=('微软雅黑', 14))],
            [sg.Text('Batch Size', font=('微软雅黑', 14)), sg.Input('4', key='-BATCH_SIZE-', size=8, font=('微软雅黑', 14))],
            [sg.Checkbox('使用BN', True, key='-USE_BN-', font=('微软雅黑',14))],
            [sg.Button('应用参数', key='-APPLY_PARAMS-', size=(12, 2), font=('微软雅黑', 14))],
            [sg.Button('保存最优权重', key='-SAVE_BEST-', size=(14, 2), font=('微软雅黑', 14))]
        ]
        # 监控区
        train_monitor = [
            [sg.Text("训练进度", font=('微软雅黑', 18))],
            [sg.Canvas(key=canvas_key, size=(800, 500))],
            [sg.ProgressBar(100, orientation='h', size=(40, 20), key=progress_key, bar_color=('green', 'white'))],
            [sg.Text("当前轮次:", font=('微软雅黑', 14)), sg.Text("0", key=status_keys[0], font=('微软雅黑', 14)),
             sg.Text("最佳损失:", font=('微软雅黑', 14)), sg.Text("--", key=status_keys[1], font=('微软雅黑', 14))]
        ]
        # 数据准备区
        def data_prep_layout():
            return [
                [sg.Text('数据准备区', font=('微软雅黑', 16))],
                [sg.Text('DICOM目录', font=('微软雅黑', 12))],
                [sg.Input(key='-DICOM_DIR-', font=('微软雅黑', 12)), sg.FolderBrowse('选择DICOM文件夹', font=('微软雅黑', 12))],
                [sg.Text('NII输出目录', font=('微软雅黑', 12))],
                [sg.Input(key='-NII_DIR-', font=('微软雅黑', 12)), sg.FolderBrowse('选择NII输出文件夹', font=('微软雅黑', 12))],
                [sg.Button('开始转换', key='-CONVERT_DICOM-', size=(12, 2), font=('微软雅黑',14))],
                [sg.HSeparator()],
                [sg.Text("数据集划分比例 (训练/验证/测试)", font=('微软雅黑', 12))],
                [sg.Input('70', size=5, font=('微软雅黑', 12)), sg.Input('20', size=5, font=('微软雅黑', 12)), sg.Input('10', size=5, font=('微软雅黑', 12))],
                [sg.Button('生成数据集', key='-SPLIT_DATA-', size=(12, 2), font=('微软雅黑',14))]
            ]
        # 新增：模型测试区
        def test_panel_layout():
            return [
                [sg.Text('模型测试', font=('微软雅黑', 16))],
                [sg.Text('测试集目录', font=('微软雅黑', 12)), sg.Input(key='-TEST_DATA-', font=('微软雅黑', 12)), sg.FolderBrowse('导入测试集', font=('微软雅黑', 12))],
                [sg.Text('选择模型', font=('微软雅黑', 12)), sg.Combo(['UNet3Plus'], default_value='UNet3Plus', key='-MODEL_SELECT-', font=('微软雅黑', 12), readonly=True)],
                [sg.Button('开始分割', key='-START_SEG-', size=(12, 2), font=('微软雅黑', 14))],
                [sg.Text('导出目录', font=('微软雅黑', 12)), sg.Input(key='-EXPORT_DIR-', font=('微软雅黑', 12)), sg.FolderBrowse('选择导出文件夹', font=('微软雅黑', 12))],
                [sg.Button('导出分割结果', key='-EXPORT_RESULT-', size=(16, 2), font=('微软雅黑', 14))]
            ]
        # 主布局
        layout = [
            [sg.TabGroup([[ 
                sg.Tab('数据准备', data_prep_layout()),
                sg.Tab('模型训练', [[
                    sg.Column(param_layout, vertical_alignment='top', pad=(10,10)),
                    sg.VSeparator(),
                    sg.Column(train_monitor, vertical_alignment='top', pad=(10,10))
                ]]),
                sg.Tab('模型测试', test_panel_layout())  # 新增Tab
            ]], key='-TABGROUP-', enable_events=True, expand_x=True, expand_y=True)],
            [sg.StatusBar("就绪", key='-STATUS-', font=('微软雅黑',14), size=(60, 1)), 
             sg.Push(), 
             sg.Button('开始训练', key='-START_TRAIN-', size=(14, 2), font=('微软雅黑', 14))]
        ]
        self.window = sg.Window(
            "MedUNet3+", layout, finalize=True, size=(1200, 800), resizable=True, element_justification='center', use_default_focus=False
        )
        self.monitor = TrainingMonitor(canvas_key, progress_key, status_keys)
        self.monitor.attach_to_window(self.window)

    def event_loop(self):
        """主事件循环"""
        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED:
                break
            elif event == '-SAVE_BEST-':
                self.model_mgr.save_checkpoint('best_model_manual', is_best=True)
                self.window.window['-STATUS-'].update('已手动保存当前最优权重')
            # 处理其他事件...
        self.window.close()

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