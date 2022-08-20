# KGB-DMOEA

A Python implementation of the **K**nowledge **G**uided **B**ayesian **D**ynamic **M**ulti-**O**bjective **E**volutionary **A**lgorithm (KGB-DMOEA).

## Description

The algorithm is implemented based on [1]. KGB-DMOEA saves historical Pareto-Optimal Solutions in an archive. When environmental change is detected, a knowledge reconstruction-examination strategy (KRE) is conducted in order to divide historical optimal solutions into useful and useless solutions for the current environment. Subsequently a naive Bayesian classifier is trained by using the classified historical solutions as samples. The trained classifier is able to predict from a set of randomly generated solutions, which ones are useful to the new environment. This results in a predicted population for the new environment that can then be optimized using any population based algorithm.

The implementation uses [Pymoo](https://github.com/anyoptimization/pymoo) as a framework [2].

## Status

- Currently NSGA-2 is used a static optimizer [4].
- The implementation is still in development and has been tested on the DF Problem Suite so far, but further validation is needed [3].

## Support and contributions

Feel free to contact Charles David Mupende s4chmupe@uni-trier.de if you have any questions or suggestions in regards to the implementation. For question concerning the algorithm, please contact the original authors.

## References

[1] Ye, Yulong, Lingjie Li, Qiuzhen Lin, Ka-Chun Wong, Jianqiang Li, and Zhong Ming. “Knowledge Guided Bayesian Classification for Dynamic Multi-Objective Optimization.” Knowledge-Based Systems, June 2022, 109173 <https://doi.org/10.1016/j.knosys.2022.109173>.

[2] J. Blank and K. Deb, pymoo: Multi-Objective Optimization in Python, in IEEE Access, vol. 8, pp. 89497-89509, 2020 <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9078759>.

[3] Shouyong Jiang, Shengxiang Yang, Xin Yao, Kay Chen Tan, Marcus Kaiser, and Natalio Krasnogor. Benchmark problems for cec2018 competition on dynamic multiobjective optimisation. In 2018 <http://homepages.cs.ncl.ac.uk/shouyong.jiang/cec2018/CEC2018_Tech_Rep_DMOP.pdf>.

[4] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan. A fast and elitist multiobjective genetic algorithm: nsga-II. Trans. Evol. Comp, 6(2):182–197, April 2002 <http://dx.doi.org/10.1109/4235.996017>.
