Gemm and storage order
======================

`Gemm <https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3>`_
means general matrix multiplication. It is a common routine in linear algebra.


.. math::

    Gemm(A, B, C, tA, tB, \alpha, \beta) = \alpha A^{tA} B^{tB} + \beta C

Where :math:`A^{tA}` means *A* is *tA* is 0 and :math:`A'` if *tA* is 1.
The coefficients of a matrix are stored in memory in a one dimension
array *T*: :math:`A(i,j) = T[i * C + j]` where *C* is the number of columns
of matrix A. In that case, the storage is said as *row major*. In case
:math:`A(i,j) = T[j * R + i]` where *R* is the number of rows,
the storage is *column major*.

We define a matrix *A* with :math:`(I, J, M, R)`, it has *I*
rows, *J* columns, the memory buffer is *M* and the matrix order
*R*. In that case, we can express the transpose of this matrix by:
If :math:`A=(I,J,M,R)`, then :math:`A' = (J,I,M,C)`.

Let's use that notation for :math:`A=(I,J,M_A,R)`, :math:`B=(J,K,M_B,R)`
and :math:`C=(I,K,M_C,R)`. We note :math:`D =  A^{tA} B^{tB} = (I, K, M_D, R)`.

.. math::

    \begin{array}{rcl}
    \alpha A^{tA} B^{tB} + \beta C &=& \alpha (I,J,M_A,R)^{tA} (J,K,M_B,R)^{tB} + \beta (I,K,M_C,R) \\
    &=& \left( \alpha (I,J,M_A,R)^{tA} (J,K,M_B,R)^{tB} + \beta (I,K,M_C,R) \right)'' \\
    &=& \left( (J,K,M_B,R)^{1-tB} \alpha (I,J,M_A,R)^{1-tA}  + \beta (I,K,M_C,R)' \right)' \\
    &=& \left( (K,J,M_B,C)^{tB} \alpha (J,I,M_A,C)^{tA}  + \beta (K,I,M_C,C) \right)' (*)\\
    &=& \left( (K,I,M_D,C)  + \beta (K,I,M_C,C) \right)' \\
    &=&  (I,K,M_D,R)  + \beta (I,K,M_C,R) 
    \end{array}

This trick can be used to run the computation of matrices using
a column major algorithm instread of a row major algorithm
by using line `(*)` as a replacement.
