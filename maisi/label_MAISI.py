import pandas as pd

MAISI_labels = {
    "liver": 1,
    "dummy1": 2,
    "spleen": 3,
    "pancreas": 4,
    "right kidney": 5,
    "aorta": 6,
    "inferior vena cava": 7,
    "right adrenal gland": 8,
    "left adrenal gland": 9,
    "gallbladder": 10,
    "esophagus": 11,
    "stomach": 12,
    "duodenum": 13,
    "left kidney": 14,
    "bladder": 15,
    "dummy2": 16,
    "portal vein and splenic vein": 17,
    "dummy3": 18,
    "small bowel": 19,
    "dummy4": 20,
    "dummy5": 21,
    "brain": 22,
    "lung tumor": 23,
    "pancreatic tumor": 24,
    "hepatic vessel": 25,
    "hepatic tumor": 26,
    "colon cancer primaries": 27,
    "left lung upper lobe": 28,
    "left lung lower lobe": 29,
    "right lung upper lobe": 30,
    "right lung middle lobe": 31,
    "right lung lower lobe": 32,
    "vertebrae L5": 33,
    "vertebrae L4": 34,
    "vertebrae L3": 35,
    "vertebrae L2": 36,
    "vertebrae L1": 37,
    "vertebrae T12": 38,
    "vertebrae T11": 39,
    "vertebrae T10": 40,
    "vertebrae T9": 41,
    "vertebrae T8": 42,
    "vertebrae T7": 43,
    "vertebrae T6": 44,
    "vertebrae T5": 45,
    "vertebrae T4": 46,
    "vertebrae T3": 47,
    "vertebrae T2": 48,
    "vertebrae T1": 49,
    "vertebrae C7": 50,
    "vertebrae C6": 51,
    "vertebrae C5": 52,
    "vertebrae C4": 53,
    "vertebrae C3": 54,
    "vertebrae C2": 55,
    "vertebrae C1": 56,
    "trachea": 57,
    "left iliac artery": 58,
    "right iliac artery": 59,
    "left iliac vena": 60,
    "right iliac vena": 61,
    "colon": 62,
    "left rib 1": 63,
    "left rib 2": 64,
    "left rib 3": 65,
    "left rib 4": 66,
    "left rib 5": 67,
    "left rib 6": 68,
    "left rib 7": 69,
    "left rib 8": 70,
    "left rib 9": 71,
    "left rib 10": 72,
    "left rib 11": 73,
    "left rib 12": 74,
    "right rib 1": 75,
    "right rib 2": 76,
    "right rib 3": 77,
    "right rib 4": 78,
    "right rib 5": 79,
    "right rib 6": 80,
    "right rib 7": 81,
    "right rib 8": 82,
    "right rib 9": 83,
    "right rib 10": 84,
    "right rib 11": 85,
    "right rib 12": 86,
    "left humerus": 87,
    "right humerus": 88,
    "left scapula": 89,
    "right scapula": 90,
    "left clavicula": 91,
    "right clavicula": 92,
    "left femur": 93,
    "right femur": 94,
    "left hip": 95,
    "right hip": 96,
    "sacrum": 97,
    "left gluteus maximus": 98,
    "right gluteus maximus": 99,
    "left gluteus medius": 100,
    "right gluteus medius": 101,
    "left gluteus minimus": 102,
    "right gluteus minimus": 103,
    "left autochthon": 104,
    "right autochthon": 105,
    "left iliopsoas": 106,
    "right iliopsoas": 107,
    "left atrial appendage": 108,
    "brachiocephalic trunk": 109,
    "left brachiocephalic vein": 110,
    "right brachiocephalic vein": 111,
    "left common carotid artery": 112,
    "right common carotid artery": 113,
    "costal cartilages": 114,
    "heart": 115,
    "left kidney cyst": 116,
    "right kidney cyst": 117,
    "prostate": 118,
    "pulmonary vein": 119,
    "skull": 120,
    "spinal cord": 121,
    "sternum": 122,
    "left subclavian artery": 123,
    "right subclavian artery": 124,
    "superior vena cava": 125,
    "thyroid gland": 126,
    "vertebrae S1": 127,
    "bone lesion": 128,
    "dummy6": 129,
    "dummy7": 130,
    "dummy8": 131,
    "airway": 132
}

def create_maisi_excel(maisi_labels):
    # Create a list of dictionaries for the DataFrame
    data = [
        {
            'Index': idx,
            'Label': label,
            'Category': get_anatomical_category(label),
            'Side': get_anatomical_side(label),
            'Type': get_label_type(label)
        }
        for label, idx in maisi_labels.items()
    ]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by Index
    df = df.sort_values('Index')
    
    # Create Excel writer
    with pd.ExcelWriter('MAISI_labels.xlsx') as writer:
        # Write main sheet
        df.to_excel(writer, sheet_name='All Labels', index=False)
        
        # Create category summary
        category_summary = df['Category'].value_counts().reset_index()
        category_summary.columns = ['Category', 'Count']
        category_summary.to_excel(writer, sheet_name='Category Summary', index=False)
        
        # Create side summary
        side_summary = df['Side'].value_counts().reset_index()
        side_summary.columns = ['Side', 'Count']
        side_summary.to_excel(writer, sheet_name='Side Summary', index=False)
        
        # Create type summary
        type_summary = df['Type'].value_counts().reset_index()
        type_summary.columns = ['Type', 'Count']
        type_summary.to_excel(writer, sheet_name='Type Summary', index=False)

def get_anatomical_category(label):
    """Determine anatomical category based on label name."""
    if any(bone in label for bone in ['vertebrae', 'rib', 'sacrum', 'skull', 'sternum', 'hip', 'femur', 'humerus', 'scapula', 'clavicula']):
        return 'Skeletal System'
    elif any(organ in label for bone in ['lung', 'trachea', 'airway']):
        return 'Respiratory System'
    elif any(organ in label for bone in ['heart', 'aorta', 'vena', 'artery', 'vein']):
        return 'Cardiovascular System'
    elif any(organ in label for bone in ['liver', 'gallbladder', 'pancreas', 'stomach', 'small bowel', 'duodenum', 'colon']):
        return 'Digestive System'
    elif any(organ in label for bone in ['kidney', 'bladder', 'prostate']):
        return 'Urinary System'
    elif any(organ in label for bone in ['spleen', 'thyroid']):
        return 'Endocrine/Lymphatic System'
    elif any(muscle in label for muscle in ['gluteus', 'iliopsoas', 'autochthon']):
        return 'Muscular System'
    elif any(nerve in label for nerve in ['brain', 'spinal cord']):
        return 'Nervous System'
    elif any(tumor in label for tumor in ['tumor', 'cancer', 'lesion']):
        return 'Pathology'
    elif 'dummy' in label:
        return 'Placeholder'
    else:
        return 'Other'

def get_anatomical_side(label):
    """Determine anatomical side based on label name."""
    if 'left' in label:
        return 'Left'
    elif 'right' in label:
        return 'Right'
    else:
        return 'Midline/Bilateral'

def get_label_type(label):
    """Determine the type of label."""
    if 'dummy' in label:
        return 'Placeholder'
    elif any(pathology in label for pathology in ['tumor', 'cancer', 'lesion']):
        return 'Pathology'
    elif any(vessel in label for vessel in ['artery', 'vein', 'vena', 'vessel']):
        return 'Vasculature'
    else:
        return 'Anatomy'

# Generate the Excel file
create_maisi_excel(MAISI_labels)