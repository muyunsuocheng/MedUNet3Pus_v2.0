import numpy as np
import nibabel as nib
import SimpleITK as sitk
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler

class NiiPreprocessor:
    def __init__(self, target_shape=(256, 256, 128), normalize=True):
        self.target_shape = target_shape
        self.normalize = normalize
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_nii(self, filepath):
        """加载NIfTI文件并返回numpy数组和元数据"""
        img = nib.load(filepath)
        data = img.get_fdata()
        affine = img.affine
        header = img.header
        return data, affine, header

    def resample_image(self, data, original_spacing, new_spacing=(1.0, 1.0, 1.0)):
        """重采样图像到统一间距"""
        sitk_image = sitk.GetImageFromArray(data)
        sitk_image.SetSpacing(original_spacing[::-1])  # NIfTI的轴顺序需要反转

        new_size = [int(np.round(old_sz*(old_spc/new_spc))) 
                   for old_sz, old_spc, new_spc in zip(data.shape, original_spacing, new_spacing)]
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetInterpolator(sitk.sitkLinear)
        return sitk.GetArrayFromImage(resampler.Execute(sitk_image))

    def preprocess(self, data, affine):
        """完整预处理流程"""
        # 重采样
        original_spacing = affine[:3, :3].diagonal()
        resampled = self.resample_image(data, original_spacing)
        
        # 标准化
        if self.normalize:
            normalized = self.scaler.fit_transform(resampled.reshape(-1, 1)).reshape(resampled.shape)
        else:
            normalized = resampled
            
        # 调整尺寸
        processed = resize(normalized, self.target_shape, mode='constant', preserve_range=True)
        return processed.astype(np.float32)

    def save_processed(self, data, output_path, affine):
        """保存预处理后的NIfTI文件"""
        img = nib.Nifti1Image(data, affine)
        nib.save(img, output_path)

    @staticmethod
    def split_volume(data, slice_axis=2):
        """将3D体数据分割为2D切片"""
        return np.split(data, data.shape[slice_axis], axis=slice_axis)