#!/bin/bash
clear
echo "--ruff--"
ruff .
echo "--cython-lint--"
cython-lint .
echo "--clang-format--"
find onnx_extended -type f \( -name "*.h" -o -name "*.hpp" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.cc" -o -name "*.cu" \) | while read f; do
    echo "Processing '$f'";
    clang-format --length 88 -i $f;
done
echo "--cmake-lint--"
find cmake -type f \( -name "*.cmake" -o -name "*.txt" \) | while read f; do
    echo "Processing '$f'";
    cmake-lint $f --line-width=88 --disabled-codes C0103 C0113;
done
