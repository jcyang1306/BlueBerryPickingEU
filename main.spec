# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=['/usr/local/lib/', 
            '/usr/lib/aarch64-linux-gnu/',
            '/usr/local/lib/python3.10/dist-packages/open3d/',
            '/usr/local/lib/python3.10/dist-packages/'
            ],
    binaries=[('eu_arm/lib/linux_arm64','.')],
    datas=[('/usr/local/lib/python3.10/dist-packages/cuda/bindings/', 'cuda/bindings'),
           ('/home/hkclr/.local/lib/python3.10/site-packages/open3d/', '.'),
           ('env_cfg.json', '.')  
            ],

    hiddenimports=['cuda-cudart', 'cuda-bindings'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe1 = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe1,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
