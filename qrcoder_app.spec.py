# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the QRcoder desktop app (one-directory build)."""

import os

block_cipher = None
root = os.path.abspath(SPECPATH)

hiddenimports = [
    "zxingcpp",
    "pyzbar",
    "segno",
    "qrcode",
    "cv2",
    "numpy",
    "PIL",
]

a = Analysis(
    ["main.py"],
    pathex=[root],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "jedi", "pytest"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="QRcoder",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="QRcoder",
)
