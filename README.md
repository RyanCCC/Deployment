# Deployment

部署深度学习应用


## CMakeListDemo

CmakeList使用Demo，包括生成执行文件以及生成DLL的Demo。在此讲一下DLL，后续会将算法模型编译成dll供程序调用。

DLL可以将程序模块化为单独的组件，可参考微软官方文档：[dynamic link library](https://docs.microsoft.com/zh-cn/troubleshoot/windows-client/deployment/dynamic-link-library)，DLL具有如下优势：

- 使用更少资源。当多个程序使用的函数库时，DLL可以减少在磁盘和物理内存中加载的代码重复。它不仅会太大影响前台运行的程序性能，还会影响在Windows操作系统上运行的其他程序的性能。
- 提升模块化体系结构。DLL有助于推动开发模块化程序。
- 简化部署和安装。

DLL编译过程：注意根目录指的是`CMakeListDemo\dllDemo\`
1. 在下运行`cmake`命令，在根目录和lib目录下编译出`Makefile`文件
2. 在根目录或者lib目录下使用`make install`即可编译出`DLL`库
3. 在根目录的`lib_out`下生成了DLL，名称为`testdll.dll`

