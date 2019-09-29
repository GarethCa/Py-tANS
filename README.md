# Py-tANS

This repository contains an implementation of the tANS algorithm developer by both Yann Collet and Jarek Duda.

Asymmetric Numeral Systems is an approach to entropy encoding discovered by Jarek Duda, detailed in the following report. https://arxiv.org/abs/1311.2540 

The code in this repository is an adaptation of the method of tANS used by Yann Collet within Zstd, developed for Facebook, specifically the Finite State Entropy code. The original code detailed its usage with the C programming language while this repo contains a Python Implementation.

## Algorithm Overview

<img src=https://3.bp.blogspot.com/-2kAzQkAjifA/WXFdxA2UjYI/AAAAAAAABwo/YcQwCC7Jmm0p8x55y-d3tuBwxdk-MtnrgCPcBGAYYCw/s1600/4states.png />

<img src=https://3.bp.blogspot.com/-4fGLCD4S3ck/WXI2nGBruzI/AAAAAAAABww/pXffRq9_TT4IdHTKSqupSKRhGyeIEZEXgCLcBGAs/s1600/5ranges.png />

<img src=http://2.bp.blogspot.com/-xbkXS6jDSCk/Uvf238sQKmI/AAAAAAAAA-Q/I0AmHbver98/s1600/16states_fastScan.png />

<img src=http://1.bp.blogspot.com/-qKIrVODaV6s/Uv3nbEyB8ZI/AAAAAAAAA_M/4SyWsR-Ws9U/s1600/5_16_linked_destinations.png />