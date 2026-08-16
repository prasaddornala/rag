import shutil
from pathlib import Path
import sys

root = Path(__file__).parent
out_name = 'Week 5_Graded Mini Project_dornala'
zip_path = root / (out_name + '.zip')

# files required by Submission_Guidelines
files = [
    root / 'mp1_prompt_lab.ipynb',
    root / 'results' / 'mp1_comparison.md',
    root / 'mp1_writeup.md',
    root / 'requirements.txt',
    root / 'README_SUBMISSION.md',
]

# verify files exist
missing = [str(p) for p in files if not p.exists()]
if missing:
    print('Missing files, aborting:', missing)
    sys.exit(1)

# create a temp folder structure
temp_dir = root / 'mp1_submission_temp'
if temp_dir.exists():
    shutil.rmtree(temp_dir)
temp_dir.mkdir()

# copy files
for p in files:
    dest = temp_dir / p.name
    # Rename README_SUBMISSION.md to README.md in the zip
    if p.name == 'README_SUBMISSION.md':
        dest = temp_dir / 'README.md'
    shutil.copy2(p, dest)

# make zip
shutil.make_archive(str(root / out_name), 'zip', root_dir=temp_dir)

# cleanup
shutil.rmtree(temp_dir)
print('Created', zip_path)
print('Size (bytes):', zip_path.stat().st_size)
