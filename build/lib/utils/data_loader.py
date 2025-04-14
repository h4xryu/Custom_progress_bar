import numpy as np
import wfdb
import pandas as pd
from pathlib import Path

def load_wfdb_record(record_path):
    """WFDB 형식의 ECG 레코드 로드"""
    try:
        record = wfdb.rdrecord(record_path)
        return record.p_signal[:, 0], record.fs
    except Exception as e:
        raise Exception(f"WFDB 레코드 로드 실패: {str(e)}")

def load_csv_data(file_path):
    """CSV 파일에서 ECG 데이터 로드"""
    try:
        df = pd.read_csv(file_path)
        required_columns = ['time', 'amplitude']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError("CSV 파일에 'time'과 'amplitude' 열이 필요합니다.")
            
        return df['amplitude'].values, df['time'].values
        
    except Exception as e:
        raise Exception(f"CSV 파일 로드 실패: {str(e)}")

def load_data(file_path):
    """파일 확장자에 따라 적절한 로더 선택"""
    file_path = Path(file_path)
    
    if file_path.suffix == '.csv':
        return load_csv_data(file_path)
    elif file_path.suffix in ['.dat', '.hea']:
        return load_wfdb_record(str(file_path.with_suffix('')))
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {file_path.suffix}")

def save_segments(segments, labels, output_path):
    """추출된 세그먼트와 레이블 저장"""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(output_path,
                 segments=segments,
                 labels=labels)
                 
    except Exception as e:
        raise Exception(f"세그먼트 저장 실패: {str(e)}")

def load_segments(file_path):
    """저장된 세그먼트와 레이블 로드"""
    try:
        data = np.load(file_path)
        return data['segments'], data['labels']
        
    except Exception as e:
        raise Exception(f"세그먼트 로드 실패: {str(e)}") 