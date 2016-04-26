# SMO-Chunking_for_SVM
SMOplatt.py is the implementation of SMO algorithm for SVM given by John C. Platt(1999) in [Fast Training of Support Vector Machines using Sequential Minimal Optimization](http://research.microsoft.com/pubs/68391/smo-book.pdf). Here, I have written my own optimized dot product function as we can't use numpy for dot product when the length of the vectors is too large.

SMOplattSKlearn.py is same as above except that here SKlearn library is used for loading into sparse CSR matrix, which is computationally faster to work on.


Chunking.py is the implementation of Chunking algorithm for SVM. For reference one can read [Support Vector Machine Solvers](http://leon.bottou.org/publications/pdf/lin-2006.pdf) by Bottou et al.(2007). Here, I have written my own optimized dot product function as we can't use numpy for dot product when the length of the vectors is too large.

ChunkingSKlearn.py is same as above except that here SKlearn library is used for loading into sparse CSR matrix, which is computationally faster to work on.
