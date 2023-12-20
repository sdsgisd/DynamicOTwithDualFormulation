# Dynamic optimal transport with the dual formulation

This is an implementation of the discretization of dynamic optimal transport introduced in the article,

A. **Quantitative convergence of a discretization of dynamic optimal transport using the dual formulation**,  
&emsp; Sadashige Ishida and Hugo Lavenant.
[[arXiv]](https://arxiv.org/abs/2312.12213)
<p>
<img src="https://sadashigeishida.bitbucket.io/dynamic_dual_OT/test2_NX16_nobar.png" height="170px">
<img src="https://sadashigeishida.bitbucket.io/dynamic_dual_OT/test2_NX512_nobar.png" height="170px">
<img src="https://sadashigeishida.bitbucket.io/dynamic_dual_OT/test2_GT_bar.png" height="170px">
</p>
  
This repository also contains an example implementation of the discretization introduced in the article,  

B. **Optimal Transport with Proximal Splitting**,  
&emsp; Nicolas Papadakis, Gabriel Peyré, Edouard Oudet.
*SIAM Journal on Imaging Sciences* [[HAL repository]](https://epubs.siam.org/doi/10.1137/130920058)

Dependencies:  
- Code for A is written in Jupyter (Python 3) and requires the following python libraries: jupyter, numpy, scipy, matplotlib.  
- Code for B is written in Julia and requires the following Julia packages: LinearAlgebra, SparseArrays, TimerOutputs, QuadGK, DelimitedFiles.

Authors: Sadashige Ishida (code for A) and Hugo Lavenant (code for B).
