# Hybrid-Feature-Selection-using-Correlation-Clustering-and-PSO
Applying evolutionary algorithms to feature selection issues in high-dimensional spaces has proven challenging due to the ”curse of dimensionality” and the high computing cost. Our project implements and then tries to extend the HFS-CC-PSP, a three-phase hybrid feature selection algorithm. The approach simultaneously tackles processing cost and dimensionality problems by combining correlation-guided clustering and particle swarm optimization (PSO). The HFS-CC-PSO algorithm combines three distinct feature selection techniques, each with benefits. The search space is condensed in the first and second phases using a filter and a clustering-based method. The ideal feature subset is located in the third phase using an evolutionary method with global searchability. The algorithm also incorporates a rapid correlation-guided feature selection approach, a symmetric uncertainty-based feature deletion method, and other features to enhance the performance of each phase.

## Installation
You could install the package using
```bash
pip install HFS-CC-PSO
```
alternatively, you could also visit 
https://pypi.org/project/HFS-CC-PSO/

## Instructions to use the package

```python
from HFS_CC_PSO import HFS_CC_PSO
fs = HFS_CC_PSO(X, y)
```
**Note**: We recommend normalizing the data first and then passing it as an argument to the class. Our Current package version only works well if the data is normalized. Also, pass both the X and y data as pandas dataframe. Following that, you can use the code snippet below and get the reduced feature subset.

```python
final_features = fs.fit()
```

## References
Song XF, Zhang Y, Gong DW, Gao XZ. A Fast Hybrid Feature Selection Based on Correlation-Guided Clustering and Particle Swarm Optimization for High-Dimensional Data. IEEE Trans Cybern. 2022 Sep;52(9):9573-9586. doi: 10.1109/TCYB.2021.3061152. Epub 2022 Aug 18. PMID: 33729976.

## Authors/Contributors
Jignesh Nagda, Dinesh Mannari, Abhishek Singh, and Jait Mahavarkar
