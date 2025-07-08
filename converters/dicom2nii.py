import os
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from datetime import datetime
from tqdm import tqdm

class DicomConverter:
    def __init__(self):
        self.progress_callback = None
    
    def convert_series(self, input_dir, output_dir):
        reader = sitk.ImageSeriesReader()
        dicom_series = reader.GetGDCMSeriesIDs(input_dir)
        
        for series_id in tqdm(dicom_series, desc="Processing DICOM series"):
            try:
                dicom_files = reader.GetGDCMSeriesFileNames(input_dir, series_id)
                reader.SetFileNames(dicom_files)
                image = reader.Execute()
                
                # 转换坐标系为RAS
                image = sitk.DICOMOrient(image, "RAS")
                
                # 转换为NIfTI
                nii_img = sitk.GetArrayFromImage(image)
                affine = np.eye(4)
                nii = nib.Nifti1Image(nii_img, affine)
                
                # 保存文件
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                output_path = os.path.join(output_dir, f"series_{series_id}_{timestamp}.nii.gz")
                nib.save(nii, output_path)
                
                if self.progress_callback:
                    self.progress_callback(1)
                    
            except Exception as e:
                print(f"Error converting series {series_id}: {str(e)}")
                continue

class DatasetSplitter:
    @staticmethod
    def split_dataset(data_dir, ratios=(0.7, 0.2, 0.1)):
        from sklearn.model_selection import train_test_split
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
        train_val, test = train_test_split(all_files, test_size=ratios[2])
        train, val = train_test_split(train_val, test_size=ratios[1]/(1-ratios[2]))
        return train, val, test