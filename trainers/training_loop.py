import torch
from tqdm import tqdm
import numpy as np

# 训练器类
# 该类负责训练和验证模型

# trainers/training_loop.py

# trainers/training_loop.py

class Trainer:
    def __init__(self, model, device, criterion, optimizer, scheduler=None, callback=None):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_loss = float('inf')
        self.callback = callback  # 新增回调函数

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs = batch['image'].float().to(self.device)
            labels = batch['label'].long().to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        return epoch_loss

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # 调用回调函数传递训练数据
            if self.callback:
                self.callback({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_loss': self.best_loss
                })
            
            # 更新最佳损失
            if val_loss < self.best_loss:
                self.best_loss = val_loss

    def train_loop(self):
        num_epochs = getattr(self, 'train_epochs', 10)
        batch_size = getattr(self, 'batch_size', 4)
        best_loss = float('inf')
        best_model_path = 'best_model.pth'
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch(epoch, batch_size)
            val_loss = self.validate_one_epoch(epoch, batch_size)
            # 更新监控器
            self.window.monitor.update_metrics(
                train_loss=train_loss,
                val_loss=val_loss,
                current_epoch=epoch,
                total_epochs=num_epochs,
                best_loss=min(self.window.monitor.val_loss) if self.window.monitor.val_loss else None
            )
            # 保存最优权重
            if val_loss < best_loss:
                best_loss = val_loss
                self.model_mgr.save_checkpoint('best_model', epoch=epoch, is_best=True)
            self.window.window.write_event_value('-STATUS_UPDATE-', f'训练进度: {epoch}/{num_epochs}')
        self.is_training = False
        self.window.window.write_event_value('-STATUS_UPDATE-', '训练完成')