#!/bin/bash

find onnx_extended -type f \( -name "*.h" -o -name "*.hpp" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.cc" -o -name "*.cu" \) | while read f; do
    echo "Processing '$f'";
    clang-format --length 88 -i $f;
done
