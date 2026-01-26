import os, csv
import cv2
import pydicom
from tqdm import tqdm

def get_pixel_spacing(dicom_dir: str):
    dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    dicom_files.sort()
    dicom_path = os.path.join(dicom_dir, dicom_files[0])
    ds = pydicom.dcmread(dicom_path)
    spacing = ds.PixelSpacing  # [row_spacing, col_spacing]
    slice_thickness = ds.SliceThickness
    return float(spacing[0]), float(spacing[1]), float(slice_thickness)


def counting(working_dir: str, predicted_dir: str, output_csv: str = None):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_dict = {}

    if not os.path.exists(output_csv):
        header = ['Patient ID', 'Date', 'Series', 'Total Pixel Count', 'Total Size (cm^2)', 'Total Volume (cm^3)']
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
    else:
        with open(output_csv, mode='r', newline='') as file:
            reader = csv.reader(file)
            existing_rows = list(reader)
            for idx, row in enumerate(existing_rows[1:]):
              key = f"{row[0]}_{row[1]}_{row[2]}"
              result_dict[key] = idx + 1
    
    patient_list = [d for d in os.listdir(predicted_dir) if os.path.isdir(os.path.join(predicted_dir, d))]
    patient_list.sort()
    for patient in tqdm(patient_list, desc="Counting pixels"):
        date_list = [d for d in os.listdir(os.path.join(predicted_dir, patient)) if os.path.isdir(os.path.join(predicted_dir, patient, d))]
        date_list.sort()
        for date in date_list:
            series_list = [d for d in os.listdir(os.path.join(predicted_dir, patient, date)) if os.path.isdir(os.path.join(predicted_dir, patient, date, d))]
            series_list.sort()
            for series in series_list:
                cur_pred_dir = os.path.join(predicted_dir, patient, date, series)
                cur_pred_image_dir = os.path.join(predicted_dir, patient, date, series, "pred")
                w, h, t = get_pixel_spacing(os.path.join(working_dir, patient, date, series)) # mm
                
                pixel_size = w * h  # mm^2
                pixel_volume = pixel_size * t  # mm^3
                pxile_size_cm2 = pixel_size / 100.0  # cm^2
                pixel_volume_cm3 = pixel_volume / 1000.0  # cm^3

                predicted_list = [os.path.join(cur_pred_image_dir, f) for f in os.listdir(cur_pred_image_dir) if f.endswith('.png')]
                predicted_list.sort()
                
                cur_header = ['Slice Index', 'Pixel Count', 'Size(cm^2)', 'Volume (cm^3)']
                cur_data = [cur_header]

                total_pixel_count = 0
                for idx, pred_image_path in enumerate(predicted_list):
                    img = cv2.imread(pred_image_path, cv2.IMREAD_GRAYSCALE)
                    count = cv2.countNonZero(img)
                    cur_size_cm2 = count * pxile_size_cm2
                    cur_volume_cm3 = count * pixel_volume_cm3
                    total_pixel_count += count
                    cur_data.append([idx, count, f"{cur_size_cm2:.4f}", f"{cur_volume_cm3:.4f}"])
                    
                cur_csv_path = os.path.join(cur_pred_dir, "counting.csv")
                with open(cur_csv_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(cur_data)
                
                total_size_cm2 = total_pixel_count * pxile_size_cm2
                total_volume_cm3 = total_pixel_count * pixel_volume_cm3
                
                key = f"{patient}_{date}_{series}"
                if key in result_dict:
                    # Update existing row
                    rows = []
                    with open(output_csv, mode='r', newline='') as file:
                        reader = csv.reader(file)
                        rows = list(reader)
                    rows[result_dict[key]][3] = total_pixel_count
                    rows[result_dict[key]][4] = f"{total_size_cm2:.4f}"
                    rows[result_dict[key]][5] = f"{total_volume_cm3:.4f}"
                    with open(output_csv, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(rows)
                else:
                    # Append new row
                    with open(output_csv, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([patient, date, series, total_pixel_count, f"{total_size_cm2:.4f}", f"{total_volume_cm3:.4f}"])
