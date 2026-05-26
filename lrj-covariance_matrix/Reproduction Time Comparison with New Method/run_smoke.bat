@echo off
call D:\miniconda3\Scripts\activate.bat mmmd_boost
cd /d "D:\Fudan\GitHub repositories\MMMD-boost-kernel-two-sample\Reproduction Time Comparison with New Method"
Rscript --no-save smoke_test.R
