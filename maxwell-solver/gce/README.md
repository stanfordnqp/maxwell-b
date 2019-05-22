TODO
====

*   Clean-up and update documentation.


What is GCE?
============
GCE stands for Gird Compute Engine, written by Jesse Lu in early 2012.


What does GCE do?
=================
GCE makes it easy to write fast 3D finite-difference applications 
  for CUDA.


How does GCE work?
==================
GCE provides a simple interface for manipulating gridded 3D data
  on the GPU.
GCE is based on simple memory and execution objects
  which hide non-essential features and details, 
  allowing applications to be defined in a simple, abstract way.


What is GCE built on?
=====================
GCE is heavily dependent on PyCUDA.


Interface overview
==================

For a simple example of GCE at work, see test_example.py.

Space
-----
The space forms the context for colocating grids and kernels.
For example, creating two grids on the same space tells GCE that
  these two grids should be overlaid on top of each other.
In the same way, defining a kernel on the space defines which grid elements
  will be updated.
As such, the space contains all the intra-process communication elements
  needed to synchronize grids and to execute kernels in parallel.
Also, the space contains all the device information needed to run kernels
  on the GPU devices.

Currently, the creation of only one global space is supported.

Grid
----
Grids represent three-dimensional fields. 
To efficiently operate on Grids, every element in a Grid has limited visibility.
This means that when operating on a Grid (with a Kernel),
  only the certain adjacent neighboring elements may be accessed.
Specifically, only elements within a cube of length 2n+1 
  (where n, the stencil size, is specified by the user) may be accessed.

Const
-----
Consts are global constant arrays that be accessed by any element of any grid.
However, the values of Const elements may not be reliably changed,
  since such changes are not synchronized across devices.

Out
---
Outs are global scalars that are used to store the result of reduce operations
 on a space.
Currently only sum operations are supported.

Kernel
------
Kernels perform operations on Grids and 
  accept Grids, Consts, and Outs as input parameters.
Kernels perform both update and sum functions.
Additionally, Kernels automatically provide self-optimization features
  such as block_size optimization and loop-unrolling.

Writing code for Kernels
------------------------
GCE provides a number of simple conventions to make writing kernel code
  simpler:

*   Relative addressing of Grid elements. 
    Computations that must be performed at every point on a Grid can be
      described using relative indexing.
    For example, copying from one Grid to another can be performed in the 
      following way:
        x(0,0,0) = y(0,0,0);

Optimization tips
=================
*   Remember to turn ECC off via 'nvidia-smi -e 0', 
      this can later be checked using the test_pycuda_speed module.
