import ctypes

dll = ctypes.CDLL('./lib_out/testdll.dll')
a = dll.add(1, 2)
print(a)  