@echo off
call D:\miniconda3\Scripts\activate.bat mmmd_boost
cd /d "D:\Fudan\GitHub repositories\MMMD-boost-kernel-two-sample\PathMNIST"
python smoke.py
