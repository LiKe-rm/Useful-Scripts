## cpp源代码文件解析工具
### clang安装方法：
pip install clang-16.0.1.1-py3-none-any.whl
### llvm使用方法
llvm[下载地址](https://github.com/llvm/llvm-project/releases/tag/llvmorg-17.0.1) 

本例中使用的版本是`LLVM-17.0.1-win64.exe`

下载后，双击安装, 在python代码中设定对应路径即可，示例：
```python
clang.cindex.Config.set_library_file('D:/software/LLVM/bin/libclang.dll')
```
