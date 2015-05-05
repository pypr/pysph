SET DISTUTILS_USE_SDK=1
SET MSSdk=1
"C:\Program Files (x86)\Common Files\Microsoft\Visual C++ for Python\9.0\vcvarsall.bat" amd64
call %* || EXIT 1

