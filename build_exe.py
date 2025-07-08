from PyInstaller.__main__ import run

build_params = [
    'main.py',
    '--name=MedUNet3Plus',
    '--onefile',
    '--windowed',
    '--add-data=assets;assets',
    '--hidden-import=torch',
    '--hidden-import=SimpleITK._SimpleITK',
    '--hidden-import=nibabel',
    '--collect-all=nibabel',
    '--icon=assets/icon.ico'
]

if __name__ == '__main__':
    run(build_params)