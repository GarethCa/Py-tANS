# Py-tANS

This repository contains an implementation of the tANS algorithm developer by both Yann Collet and Jarek Duda.

Asymmetric Numeral Systems is an approach to entropy encoding discovered by Jarek Duda, detailed in the following report. https://arxiv.org/abs/1311.2540 

The code in this repository is an adaptation of the method of tANS used by Yann Collet within Zstd, developed for Facebook, specifically the Finite State Entropy code. The original code detailed its usage with the C programming language while this repo contains a Python Implementation. https://github.com/Cyan4973/FiniteStateEntropy  http://fastcompression.blogspot.com/2013/12/finite-state-entropy-new-breed-of.html

## Algorithm Overview

tANS or FSE is a method of range coding using an Asymmetric Numeral System. A set of states are used so that the probability of a state occuring is as close to equal as possible as the probability of the state it represents.

The action of changing states is analagous to changing state within a finite state machine, where we can output new symbols to a bit stream when we change state by encoding one more character.

<img src=https://3.bp.blogspot.com/-2kAzQkAjifA/WXFdxA2UjYI/AAAAAAAABwo/YcQwCC7Jmm0p8x55y-d3tuBwxdk-MtnrgCPcBGAYYCw/s1600/4states.png />

To encode symbols using the number of bits that we can theoretically reach by calculating the optimal bits per symbol using the Shannon Entropy calculation, is the hardest part of this algorithm to understand. 
We only output an integer number of bits for symbols which may have a fractional number of bits as their Shannon Entropy optimal value, therefore we must occasionally use fewer or more bits to encode a symbol as to best match the fractional bits used on average.

This process can be achieved by outputting a variable number of bits depending on which "subrange" of states the current state falls into. The idea is that the number of bits output will not be optimal on a one output basis, but instead will average the the correct fraction.

<img src=https://3.bp.blogspot.com/-4fGLCD4S3ck/WXI2nGBruzI/AAAAAAAABww/pXffRq9_TT4IdHTKSqupSKRhGyeIEZEXgCLcBGAs/s1600/5ranges.png />

Another factor to consider is the concept of moving between states, which correctly tell us what symbol was last used. This is accomplished by outputting the Binary represntation of how many states we must move as the bit output. This therefore requires the spacing between states to be consistent with the sub-range bits used seen in the above picture.

To acomplish this problem, we spread the states across the state space in such a way to provide the correct gaps between sub-range values. 

This spreading of symbols can be done in multiple different ways, and can even be scrambled randomly based on some cryptographic key, meaning we may not keep optimal performance but will actually provide security by only being able to decode using a correct key.

<img src=http://2.bp.blogspot.com/-xbkXS6jDSCk/Uvf238sQKmI/AAAAAAAAA-Q/I0AmHbver98/s1600/16states_fastScan.png />


## Limitations
There are issues associated with this algorithm such as the fragility of the encoding, where if one bit is corrupted in our bitstream, we may have a totally different output as a consequence.