# CUDA examples

## ADD VECTOR
Given three vectors a, b and c, save the sum of the two first vectors (a + b) into the third one (c).

Improvements:
  - Naive implementation: 508.27 us.

## MATRIX MULTIPLICATION
Given three matrix a, b and c, save the product between matrix a and b into matrix c.

Improvements:
  - Naive implementation: 329.47 ms.
  - Tiled multiplication: 59.755 ms.

## MATRIX HISTOGRAM
Given a matrix, calculate the histogram.

Improvements:
  - Naive implementation: 875.09 us.
  - First histogram of the block in shared memory: 132.71 us.

## CONV 1D
Given a vector and a kernel calculate the convolution of the vector with kernel and save it in an output vector.

Improvements:
  - Naive implementation: 423.35 us.
  - Temp varaible to save the convolution result: 364.13 us.
  - Loop unrolling: 361.08 us.
  - Kernel in constant memory: 208.85 us.

## REDUCE COL
Given a matrix, calculate the sum over each row.

Improvements:
  - Naive implementation: 154.65 us.
  - Remove stride reading in global memory: 92.309 us.