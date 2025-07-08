import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import threading
import torch
from gui.main_window import MainWindow
from models.model_manager import ModelManager
from converters.dicom2nii import DicomConverter, DatasetSplitter
import PySimpleGUI as sg
from models.UNet3Plus import UNet3Plus


def log(msg):
    print(msg)
    with open("medunet_log.txt", "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


class Application:
    def __init__(self):
        self.window = MainWindow(
            canvas_key='-TRAIN_CANVAS-',
            progress_key='-PROGRESS-',
            status_keys=('-CURRENT_EPOCH-', '-BEST_LOSS-')
        )
        self.model_mgr = ModelManager()
        self.model_mgr.register_model('UNet3Plus', UNet3Plus)  # 注册模型
        self.converter = DicomConverter()
        self.is_training = False
    
    def run(self):
        self.segmentation_results = []  # 保存分割结果
        while True:
            event, values = self.window.window.read()  
            
            if event == sg.WIN_CLOSED:
                break
                
            if event == '-CONVERT_DICOM-':
                threading.Thread(target=self.convert_dicom, args=(values['-DICOM_DIR-'], 
                                values['-NII_DIR-']), daemon=True).start()
                
            if event == '-APPLY_PARAMS-':
                self.apply_model_params(values)
                
            if event == '-START_TRAIN-':
                self.start_training()

            # 新增：模型测试相关事件
            if event == '-START_SEG-':
                test_data_dir = values['-TEST_DATA-']
                model_name = values['-MODEL_SELECT-']
                if not test_data_dir or not model_name:
                    self.window.window['-STATUS-'].update('请先选择测试集目录和模型')
                else:
                    threading.Thread(target=self.segment_test_set, args=(test_data_dir, model_name), daemon=True).start()

            if event == '-EXPORT_RESULT-':
                export_dir = values['-EXPORT_DIR-']
                # 新增：确保分割结果为非空列表
                if not self.segmentation_results or len(self.segmentation_results) == 0:
                    self.window.window['-STATUS-'].update('请先完成分割')
                elif not export_dir:
                    self.window.window['-STATUS-'].update('请选择导出目录')
                else:
                    threading.Thread(target=self.export_results, args=(export_dir,), daemon=True).start()

            if event == '-STATUS_UPDATE-':
                self.window.window['-STATUS-'].update(values[event])
        self.window.window.close()

    def convert_dicom(self, input_dir, output_dir):
        try:
            self.converter.convert_series(input_dir, output_dir)
            self.window.window.write_event_value('-STATUS_UPDATE-', '转换完成')
        except Exception as e:
            self.window.window['-STATUS-'].update(f'转换错误: {str(e)}')
    
    def apply_model_params(self, values):
        config = {
            'in_channels': int(values['-IN_CH-']),
            'out_channels': int(values['-NUM_CLASS-']),
            'feature_scale': int(values['-FEATURE_SCALE-']),
            'is_batchnorm': values['-USE_BN-'],
            'train_data': values['-TRAIN_DATA-'],
            'labels': values['-LABELS-'],
            'epochs': int(values['-EPOCHS-']),         # 新增
            'batch_size': int(values['-BATCH_SIZE-'])  # 新增
        }
        self.train_epochs = config['epochs']
        self.batch_size = config['batch_size']
        model_name = 'UNet3Plus'
        self.model_mgr.initialize_model(model_name, **config)
        self.window.window['-STATUS-'].update('模型初始化完成')
    
    def start_training(self):
        if not self.model_mgr.current_model:
            self.window.window['-STATUS-'].update('请先初始化模型')
            return
        if self.is_training:
            self.window.window['-STATUS-'].update('训练已在进行中')
            return
        self.is_training = True
        self.window.window['-STATUS-'].update('训练启动...')
        # 启动训练线程
        threading.Thread(target=self.train_loop, daemon=True).start()

    def train_loop(self):
        num_epochs = self.train_epochs  # 使用apply_model_params中保存的值
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate_one_epoch(epoch)
            self.window.monitor.update_metrics(
                train_loss=train_loss,
                val_loss=val_loss,
                current_epoch=epoch,
                total_epochs=num_epochs,
                best_loss=min(self.window.monitor.val_loss) if self.window.monitor.val_loss else None
            )
            self.window.window.write_event_value('-STATUS_UPDATE-', f'训练进度: {epoch}/{num_epochs}')
        self.is_training = False
        self.window.window.write_event_value('-STATUS_UPDATE-', '训练完成')

    def train_one_epoch(self, epoch):
        # 这里写单轮训练逻辑，返回loss
        return 1.0 / epoch  # 示例

    def validate_one_epoch(self, epoch):
        # 这里写单轮验证逻辑，返回loss
        return 1.2 / epoch  # 示例

    def segment_test_set(self, test_data_dir, model_name):
        try:
            log(f"segment_test_set被调用, test_data_dir={test_data_dir}, model_name={model_name}")
            model = self.model_mgr.get_model(model_name)
            model.eval()
            from data.dataloader import MedicalDataset
            import numpy as np
            test_dataset = MedicalDataset(test_data_dir, split='test')
            log(f"测试集样本数: {len(test_dataset)}")  # 调试
            results = []
            for idx in range(len(test_dataset)):
                sample = test_dataset[idx]
                image = sample['image']  # [1, H, W, D]
                # 假设 image shape: [1, H, W, D]
                image_np = image.squeeze(0).numpy()  # [H, W, D]
                pred_volume = []
                for d in range(image_np.shape[-1]):
                    slice_img = image_np[..., d]  # [H, W]
                    slice_tensor = torch.from_numpy(slice_img).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                    slice_tensor = slice_tensor.to(next(model.parameters()).device)  # 关键：放到模型同设备
                    with torch.no_grad():
                        output = model(slice_tensor)
                        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # [H, W]
                    pred_volume.append(pred)
                pred_volume = np.stack(pred_volume, axis=-1)  # [H, W, D]
                results.append((idx, pred_volume))
            self.segmentation_results = results
            log(f"分割结果数量: {len(self.segmentation_results)}")  # 调试
            self.window.window.write_event_value('-STATUS_UPDATE-', '分割完成，可导出结果')
        except Exception as e:
            log(f"分割异常: {e}")
            self.window.window.write_event_value('-STATUS_UPDATE-', f'分割错误: {str(e)}')

    def export_results(self, export_dir):
        try:
            import numpy as np
            import os
            import nibabel as nib
            os.makedirs(export_dir, exist_ok=True)
            affine = np.eye(4)  # 默认仿射矩阵
            for idx, pred in self.segmentation_results:
                nii_img = nib.Nifti1Image(pred.astype(np.uint8), affine)
                nib.save(nii_img, os.path.join(export_dir, f'result_{idx}.nii'))
            self.window.window.write_event_value('-STATUS_UPDATE-', '分割结果导出完成')
        except Exception as e:
            self.window.window.write_event_value('-STATUS_UPDATE-', f'导出错误: {str(e)}')

if __name__ == '__main__':
    app = Application()
    app.run()