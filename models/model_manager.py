import torch
import importlib
from pathlib import Path
from models.UNet3Plus import UNet3Plus
from gui.main_window import MainWindow
from converters.dicom2nii import DicomConverter

class ModelManager:
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = {}
        self.optimizer = None
        self.scheduler = None

    def register_model(self, name, model_class):
        """注册新模型架构"""
        self.models[name] = model_class

    def list_models(self):
        """获取可用模型列表"""
        return list(self.models.keys())



    def save_checkpoint(self, path, epoch=None, is_best=False):
        """保存完整训练状态"""
        checkpoint = {
            'model_state': self.current_model.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch,
            'config': self.model_config
        }
        suffix = "_best.pth" if is_best else f"_epoch{epoch}.pth"
        torch.save(checkpoint, Path(path).with_suffix(suffix))

    def load_checkpoint(self, path, resume_training=False):
        """加载完整训练状态"""
        checkpoint = torch.load(path, map_location=self.device)
        self.current_model.load_state_dict(checkpoint['model_state'])
        
        if resume_training and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        return checkpoint.get('epoch', 0)

    def setup_optimizer(self, optimizer_type='Adam', lr=1e-3):
        """配置优化器"""
        if optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(self.current_model.parameters(), lr=lr)
        elif optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(self.current_model.parameters(), lr=lr, momentum=0.9)
        return self.optimizer

    def setup_scheduler(self, scheduler_type='StepLR', **kwargs):
        """配置学习率调度器"""
        if not self.optimizer:
            raise RuntimeError("需要先配置优化器")
            
        if scheduler_type == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **kwargs)
        elif scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **kwargs)
        return self.scheduler

    def initialize_model(self, model_name, **config):
        """初始化模型并返回实例"""
        model_class = self.models.get(model_name)
        if model_class is None:
            raise ValueError(f"模型 {model_name} 未注册")
        self.current_model = model_class(**config).to(self.device)
        self.model_config = config
        return self.current_model

    def apply_model_params(self, values):
        config = {
            'in_channels': int(values['-IN_CH-']),
            'out_channels': int(values['-NUM_CLASS-']),
            'feature_scale': int(values['-FEATURE_SCALE-']),
            'is_batchnorm': values['-USE_BN-']  # 注意参数名
        }
        model_name = 'UNet3Plus'
        self.model_mgr.initialize_model(model_name, **config)
        self.window.window['-STATUS-'].update('模型初始化完成')

    def get_model(self, model_name):
        """
        返回已初始化的模型实例
        """
        if hasattr(self, 'current_model') and self.current_model is not None:
            return self.current_model
        else:
            raise ValueError("当前没有初始化模型")

class Application:
    def __init__(self):
        self.window = MainWindow(
            canvas_key='-TRAIN_CANVAS-',
            progress_key='-PROGRESS-',
            status_keys=('-CURRENT_EPOCH-', '-BEST_LOSS-')
        )
        self.model_mgr = ModelManager()
        self.model_mgr.register_model('UNet3Plus', UNet3Plus)  # ← 注册模型
        self.converter = DicomConverter()
        self.is_training = False