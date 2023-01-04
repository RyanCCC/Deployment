#include <windows.h>
#include <iostream>

typedef int (*add)(int, int);

int main() {
    HINSTANCE handle = LoadLibrary("./lib_out/testdll.dll");
    auto f = (add) GetProcAddress(handle, "add");
    std::cout << f(1, 32) << std::endl;
    FreeLibrary(handle);
    return 0;
}