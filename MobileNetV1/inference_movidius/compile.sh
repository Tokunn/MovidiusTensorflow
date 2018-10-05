#!/usr/bin/env sh

cd models
mvNCCompile ./mbnet_flozen.ckpt.meta -in inputs/X -on mbnet_struct/actual_softmax -s 12 -o ../../graphs/mbnet224_1.graph
cd ..
