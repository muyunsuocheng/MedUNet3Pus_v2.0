import PySimpleGUI as sg

class ParamEditor:
    def __init__(self, default_params):
        self.window = None
        self.params = default_params.copy()

    def create_layout(self):
        """创建参数编辑界面布局"""
        model_params = [
            [sg.Text('模型架构'), sg.Combo(['UNet3+', 'ResUNet3+'], key='-MODEL_TYPE-')],
            [sg.Text('输入通道'), sg.Input('1', key='-IN_CHANNELS-', size=6)],
            [sg.Text('类别数'), sg.Input('3', key='-NUM_CLASSES-', size=6)],
            [sg.Text('特征缩放'), sg.Slider((1,8), 4, orientation='h', key='-FEATURE_SCALE-')],
            [sg.Checkbox('使用批归一化', True, key='-USE_BN-')],
            [sg.Checkbox('使用深度监督', False, key='-DEEP_SUPERVISION-')]
        ]

        training_params = [
            [sg.Text('优化器'), sg.Combo(['Adam', 'SGD'], key='-OPTIMIZER-')],
            [sg.Text('初始学习率'), sg.Input('0.001', key='-LR-', size=8)],
            [sg.Text('批大小'), sg.Input('4', key='-BATCH_SIZE-', size=6)],
            [sg.Text('训练轮次'), sg.Input('100', key='-EPOCHS-', size=6)],
            [sg.Checkbox('使用早停', True, key='-EARLY_STOP-')],
            [sg.Text('早停耐心'), sg.Input('10', key='-PATIENCE-', size=6)]
        ]

        layout = [
            [sg.TabGroup([[ 
                sg.Tab('模型参数', model_params),
                sg.Tab('训练参数', training_params)
            ]])],
            [sg.Button('保存'), sg.Button('取消')]
        ]
        return layout

    def show(self):
        """显示参数编辑窗口"""
        layout = self.create_layout()
        self.window = sg.Window('参数编辑器', layout, modal=True)
        
        while True:
            event, values = self.window.read()
            if event in (None, '取消'):
                self.params = None
                break
            if event == '保存':
                self.params = {
                    'model_type': values['-MODEL_TYPE-'],
                    'in_channels': int(values['-IN_CHANNELS-']),
                    'num_classes': int(values['-NUM_CLASSES-']),
                    'feature_scale': int(values['-FEATURE_SCALE-']),
                    'use_bn': values['-USE_BN-'],
                    'deep_supervision': values['-DEEP_SUPERVISION-'],
                    'optimizer': values['-OPTIMIZER-'],
                    'lr': float(values['-LR-']),
                    'batch_size': int(values['-BATCH_SIZE-']),
                    'epochs': int(values['-EPOCHS-']),
                    'early_stop': values['-EARLY_STOP-'],
                    'patience': int(values['-PATIENCE-'])
                }
                break
        
        self.window.close()
        return self.params

# 使用示例
if __name__ == '__main__':
    default = {
        'model_type': 'UNet3+',
        'in_channels': 1,
        'num_classes': 3,
        'feature_scale': 4,
        'use_bn': True,
        'deep_supervision': False,
        'optimizer': 'Adam',
        'lr': 0.001,
        'batch_size': 4,
        'epochs': 100,
        'early_stop': True,
        'patience': 10
    }
    editor = ParamEditor(default)
    params = editor.show()
    print("最终参数:", params)