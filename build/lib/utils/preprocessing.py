import numpy as np

def label2index(label):
    """
    NSVFQECG 클래스 레이블을 인덱스로 변환합니다.
    
    Parameters:
    -----------
    label : str
        변환할 레이블 ('N', 'S', 'V', 'F', 'Q')
        
    Returns:
    --------
    int
        레이블에 해당하는 인덱스 (0-4)
    """
    label_map = {
        'N': 0,  # Normal beats
        'S': 1,  # Supraventricular premature beats
        'V': 2,  # Ventricular premature beats
        'F': 3,  # Fusion beats
        'Q': 4   # Unknown beats
    }
    return label_map.get(label, 4)  # 알 수 없는 레이블은 'Q'(4)로 간주

def index2label(index):
    """
    인덱스를 NSVFQECG 클래스 레이블로 변환합니다.
    
    Parameters:
    -----------
    index : int
        변환할 인덱스 (0-4)
        
    Returns:
    --------
    str
        인덱스에 해당하는 레이블 ('N', 'S', 'V', 'F', 'Q')
    """
    label_map = {
        0: 'N',  # Normal beats
        1: 'S',  # Supraventricular premature beats
        2: 'V',  # Ventricular premature beats
        3: 'F',  # Fusion beats
        4: 'Q'   # Unknown beats
    }
    return label_map.get(index, 'Q')  # 알 수 없는 인덱스는 'Q'로 간주