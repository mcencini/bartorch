# Writes the bytes of IN into OUT as a C array named NAME.
#
#   cmake -DIN=<file> -DOUT=<header> -DNAME=<identifier> -P embed.cmake
file(READ "${IN}" _hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _bytes "${_hex}")
file(WRITE "${OUT}"
    "/* Generated at build time; see CMakeLists.txt. */\n"
    "__attribute__((aligned(16))) static const unsigned char ${NAME}[] = {${_bytes}};\n")
