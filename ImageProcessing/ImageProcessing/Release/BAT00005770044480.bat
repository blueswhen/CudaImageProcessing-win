@echo off
 echo "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\bin\nvcc.exe"     -gencode=arch=compute_20,code=\"sm_21,compute_20\" -gencode=arch=compute_20,code=\"sm_21,compute_20\" -gencode=arch=compute_20,code=\"sm_21,compute_20\" --machine 32 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin"    -Xcompiler "/EHsc /W3 /nologo /O2 /Zi   /MT  " -I"G:\labsync\code\ImageProcessing\\ImageProcessing" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\include" -maxrregcount=0   --compile -o "Release/canny_and_edge_connection.cu.obj" canny_and_edge_connection.cu  

 cd "Release" 
 findstr /L /I "\"Release/canny_and_edge_connection.cu.obj\"" "ImageProcessing.device-link.options" >nul 2>&1 
 IF ERRORLEVEL 1 echo "Release/canny_and_edge_connection.cu.obj">> "ImageProcessing.device-link.options" 
 cd "g:\labsync\code\ImageProcessing\ImageProcessing\" 
 "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\bin\nvcc.exe"     -gencode=arch=compute_20,code=\"sm_21,compute_20\" -gencode=arch=compute_20,code=\"sm_21,compute_20\" -gencode=arch=compute_20,code=\"sm_21,compute_20\" --machine 32 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin"    -Xcompiler "/EHsc /W3 /nologo /O2 /Zi   /MT  " -I"G:\labsync\code\ImageProcessing\\ImageProcessing" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\include" -maxrregcount=0   --compile -o "Release/canny_and_edge_connection.cu.obj" "g:\labsync\code\ImageProcessing\ImageProcessing\src\canny_and_edge_connection.cu"  
if errorlevel 1 goto VCReportError
goto VCEnd
:VCReportError
echo Project : error PRJ0019: 某个工具从以下位置返回了错误代码: "Compiling with CUDA Build Rule..."
exit 1
:VCEnd