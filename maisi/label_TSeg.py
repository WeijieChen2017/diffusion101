import pandas as pd

TSv2_labels = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    16: "trachea",
    17: "thyroid_gland",
    18: "small_bowel",
    19: "duodenum",
    20: "colon",
    21: "urinary_bladder",
    22: "prostate",
    23: "kidney_cyst_left",
    24: "kidney_cyst_right",
    25: "sacrum",
    26: "vertebrae_S1",
    27: "vertebrae_L5",
    28: "vertebrae_L4",
    29: "vertebrae_L3",
    30: "vertebrae_L2",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
    44: "vertebrae_C7",
    45: "vertebrae_C6",
    46: "vertebrae_C5",
    47: "vertebrae_C4",
    48: "vertebrae_C3",
    49: "vertebrae_C2",
    50: "vertebrae_C1",
    51: "heart",
    52: "aorta",
    53: "pulmonary_vein",
    54: "brachiocephalic_trunk",
    55: "subclavian_artery_right",
    56: "subclavian_artery_left",
    57: "common_carotid_artery_right",
    58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left",
    60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left",
    62: "superior_vena_cava",
    63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein",
    65: "iliac_artery_left",
    66: "iliac_artery_right",
    67: "iliac_vena_left",
    68: "iliac_vena_right",
    69: "humerus_left",
    70: "humerus_right",
    71: "scapula_left",
    72: "scapula_right",
    73: "clavicula_left",
    74: "clavicula_right",
    75: "femur_left",
    76: "femur_right",
    77: "hip_left",
    78: "hip_right",
    79: "spinal_cord",
    80: "gluteus_maximus_left",
    81: "gluteus_maximus_right",
    82: "gluteus_medius_left",
    83: "gluteus_medius_right",
    84: "gluteus_minimus_left",
    85: "gluteus_minimus_right",
    86: "autochthon_left",
    87: "autochthon_right",
    88: "iliopsoas_left",
    89: "iliopsoas_right",
    90: "brain",
    91: "skull",
    92: "rib_left_1",
    93: "rib_left_2",
    94: "rib_left_3",
    95: "rib_left_4",
    96: "rib_left_5",
    97: "rib_left_6",
    98: "rib_left_7",
    99: "rib_left_8",
    100: "rib_left_9",
    101: "rib_left_10",
    102: "rib_left_11",
    103: "rib_left_12",
    104: "rib_right_1",
    105: "rib_right_2",
    106: "rib_right_3",
    107: "rib_right_4",
    108: "rib_right_5",
    109: "rib_right_6",
    110: "rib_right_7",
    111: "rib_right_8",
    112: "rib_right_9",
    113: "rib_right_10",
    114: "rib_right_11",
    115: "rib_right_12",
    116: "sternum",
    117: "costal_cartilages"
}

def create_tsv2_excel(tsv2_labels):
    # Create a list of dictionaries for the DataFrame
    data = [
        {
            'Index': idx,
            'Label': label,
            'Category': get_anatomical_category(label),
            'Side': get_anatomical_side(label)
        }
        for idx, label in tsv2_labels.items()
    ]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by Index
    df = df.sort_values('Index')
    
    # Create Excel writer
    with pd.ExcelWriter('TSv2_labels.xlsx') as writer:
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

def get_anatomical_category(label):
    """Determine anatomical category based on label name."""
    if any(bone in label for bone in ['vertebrae', 'rib', 'sacrum', 'skull', 'sternum', 'hip', 'femur', 'humerus', 'scapula', 'clavicula']):
        return 'Skeletal System'
    elif any(organ in label for organ in ['lung', 'trachea', 'esophagus']):
        return 'Respiratory System'
    elif any(organ in label for organ in ['heart', 'aorta', 'vena', 'artery', 'vein']):
        return 'Cardiovascular System'
    elif any(organ in label for organ in ['liver', 'gallbladder', 'pancreas', 'stomach', 'small_bowel', 'duodenum', 'colon']):
        return 'Digestive System'
    elif any(organ in label for organ in ['kidney', 'urinary_bladder', 'prostate']):
        return 'Urinary System'
    elif any(organ in label for organ in ['spleen', 'thyroid']):
        return 'Endocrine/Lymphatic System'
    elif any(muscle in label for muscle in ['gluteus', 'iliopsoas', 'autochthon']):
        return 'Muscular System'
    elif any(nerve in label for nerve in ['brain', 'spinal_cord']):
        return 'Nervous System'
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

# Generate the Excel file
create_tsv2_excel(TSv2_labels)