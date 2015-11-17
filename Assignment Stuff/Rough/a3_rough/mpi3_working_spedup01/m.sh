#!/bin/bash

mpirun -n 1 ./prg 2000 > output_1_processor.txt 
mpirun -n 2 ./prg 2000 > output_2_processor.txt
mpirun -n 4 ./prg 2000 > output_4_processor.txt
mpirun -n 8 ./prg 2000 > output_8_processor.txt
mpirun -n 12 ./prg 2000 > output_12_processor.txt
mpirun -n 16 ./prg 2000 > output_16_processor.txt
   
