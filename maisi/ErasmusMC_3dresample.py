import os
import glob
import nibabel as nib
import argparse
from tqdm import tqdm

# =============================================================================
# Configuration and Global Constants
# =============================================================================

CASE_NAMES = [
    # 'E4058',
    'E4055', 'E4058', 'E4061', 'E4066', 'E4068',
    'E4069', 'E4073', 'E4074', 'E4077', 'E4078',
    'E4079', 'E4081', 'E4084', 'E4091', 'E4092',
    'E4094', 'E4096', 'E4098', 'E4099', 'E4103',
    'E4105', 'E4106', 'E4114', 'E4115', 'E4118',
    'E4120', 'E4124', 'E4125', 'E4128', 'E4129',
    'E4130', 'E4131', 'E4134', 'E4137', 'E4138',
    'E4139',
]

# Default parameters
DEFAULT_INPUT_DIR = "ErasmusMC/HUadj"
DEFAULT_OUTPUT_DIR = "ErasmusMC/HUadj_resampled"
DEFAULT_REFERENCE_DIR = "CTAC_bed"
DEFAULT_CONTOUR_TYPE = "bcC"

def process_case(case_name, args):
    """
    Process a single case: get the HU-adjusted file and create 3dresample command.
    
    Args:
        case_name: Name of the case to process
        args: Command-line arguments
    """
    print(f"Processing case {case_name}...")
    
    # Define file paths
    input_path = os.path.join(args.input_dir, f"SynCT_{case_name}_{args.contour_type}_HUadj.nii.gz")
    output_path = os.path.join(args.output_dir, f"SynCT_{case_name}_{args.contour_type}_HUadj_resampled.nii.gz")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"  Warning: Input file not found: {input_path}")
        return
    
    # Find reference CT file for this case
    # Remove 'E' from case name to match reference file pattern
    case_id = case_name[1:]
    reference_files = glob.glob(f"{args.reference_dir}/*_{case_id}_*.nii*")
    
    if not reference_files:
        print(f"  Warning: No reference file found for case {case_name}")
        return
    
    reference_path = sorted(reference_files)[0]
    print(f"  Using reference file: {reference_path}")
    
    # Load reference file to get pixel dimensions
    reference_nifti = nib.load(reference_path)
    
    # Get pixel dimensions from the header
    dx, dy, dz = reference_nifti.header['pixdim'][1:4]
    
    # Format dimensions to 4 decimal places
    dx = f"{dx:.4f}"
    dy = f"{dy:.4f}"
    dz = f"{dz:.4f}"
    
    # Create 3dresample command
    command = f"3dresample -dxyz {dx} {dy} {dz} -prefix {output_path} -input {input_path}"
    
    # Output command
    print(f"  {command}")
    
    # If execute flag is set, run the command
    if args.execute:
        print(f"  Executing command...")
        os.system(command)
        print(f"  Command executed.")

def main():
    parser = argparse.ArgumentParser(description='Generate 3dresample commands for HU-adjusted synthetic CT images')
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR, 
                        help=f'Directory containing HU-adjusted synthetic CT images (default: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, 
                        help=f'Directory to save resampled images (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--reference_dir', type=str, default=DEFAULT_REFERENCE_DIR, 
                        help=f'Directory containing reference CT images (default: {DEFAULT_REFERENCE_DIR})')
    parser.add_argument('--contour_type', type=str, default=DEFAULT_CONTOUR_TYPE, 
                        help=f'Body contour type (default: {DEFAULT_CONTOUR_TYPE})')
    parser.add_argument('--case_id', type=str, default=None, 
                        help='Process specific case ID (default: process all cases)')
    parser.add_argument('--execute', action='store_true',
                        help='Execute the 3dresample commands (default: just print commands)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("Running with the following configuration:")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Reference directory: {args.reference_dir}")
    print(f"  Contour type: {args.contour_type}")
    print(f"  Execute commands: {args.execute}")
    
    # Determine which cases to process
    if args.case_id:
        cases_to_process = [args.case_id]
        print(f"Processing single case: {args.case_id}")
    else:
        # Process all cases in the predefined list
        cases_to_process = CASE_NAMES
        print(f"Processing all {len(cases_to_process)} cases from the predefined list")
    
    # Process each case
    for case_name in tqdm(cases_to_process):
        process_case(case_name, args)
    
    print("All processing completed.")

if __name__ == "__main__":
    main()
