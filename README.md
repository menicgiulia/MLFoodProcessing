<p align="justify">
  
# MLFoodProcessing
  
**Paper**: Machine Learning Prediction of the Degree of Food Processing

**Authors**: G. Menichetti, B. Ravandi, D. Mozaffarian, A-L. Barabasi

**DOI**: https://doi.org/10.1101/2021.05.22.21257615

### Overview of FoodProX and FPro 
![alt text](https://github.com/menicgiulia/MLFoodProcessing/blob/main/Box1.png?raw=true)


## System Requirements

### Hardware Requirements

All the codes provided require a standard computer with enough RAM to support the operations. For minimal performance, this will be a computer with about 16 GB of RAM and 4 cores. For optimal performance, we recommend a computer with the following specifications:

RAM: 64+ GB

CPU: 16+ cores

### Software Requirements

The provided codes have been tested on the following systems:

macOS (12.4) 

Windows: 7+

Software versions used:
* Python: 3.6.10
* MATLAB: 2022a
* R: 3.4.0 (2020-10-10)
  
## Installation Guide

In Python:

```
pip install pandas
pip install scipy
pip install matplotlib
pip install seaborn
pip install tqdm
pip install joblib
pip install shap
pip install rfpimp
pip install dynamicTreeCut
pip install networkx
pip install pyvis
pip install scikit-learn
pip install imbalanced-learn
```
### Version Numbers for Python Packages

* pandas: 1.1.5
* scipy: 1.5.2
* matplotlib: 3.3.4
* tqdm: 4.63.0
* joblib: 0.17.0
* shap: 0.41.0  
* rfpimp: 1.3.7  
* dynamicTreeCut: 0.1.0
* networkx: 2.5  
* pyvis: 0.3.1
* scikit-learn: 0.24.2
* imbalanced-learn: 0.8.1
  
[//]: # (In R:)

[//]: # (```)

[//]: # (install.packages&#40;c&#40;'car', 'bestNormalize', 'data.table', 'datasets', 'devtools', 'doParallel', 'foreach', 'haven', 'MASS', 'parallel', 'survival', 'zoo', 'adegenet', 'glmnet', 'corrplot&#41;&#41;)

[//]: # (```)
[//]: # (### Version Numbers for R Packages)

[//]: # ()
[//]: # (* car: 3.0.3)

[//]: # (* bestNormalize: 1.3.0)

[//]: # (* data.table: 1.11.8)

[//]: # (* datasets: 3.5.0)

[//]: # (* devtools: 2.0.1)

[//]: # (* doParallel: 1.0.14)

[//]: # (* foreach: 1.4.4)

[//]: # (* haven: 2.0.0)

[//]: # (* MASS: 7.3.51.1)

[//]: # (* parallel: 3.5.0)

[//]: # (* survival: 2.43.3)

[//]: # (* zoo: 1.8.4)

[//]: # (* adegenet: 2.1.1)

[//]: # (* glmnet: 2.0.16)

[//]: # (* corrplot: 0.84)


[//]: # (Typical install time on a normal desktop computer for all Python and R packages is less than 10 minutes.)
  


# R Environment Used in the EWAS Study


```
> sessionInfo()

R version 4.0.3 (2020-10-10)
Platform: x86_64-w64-mingw32/x64 (64-bit)
Running under: Windows 10 x64 (build 22621)

Matrix products: default

locale:
[1] LC_COLLATE=English_United States.1252 
[2] LC_CTYPE=English_United States.1252   
[3] LC_MONETARY=English_United States.1252
[4] LC_NUMERIC=C                          
[5] LC_TIME=English_United States.1252    

attached base packages:
 [1] parallel  stats4    grid      stats     graphics  grDevices utils    
 [8] datasets  methods   base     

other attached packages:
 [1] DESeq2_1.28.1               SummarizedExperiment_1.18.2
 [3] DelayedArray_0.14.1         matrixStats_0.57.0         
 [5] Biobase_2.48.0              GenomicRanges_1.40.0       
 [7] GenomeInfoDb_1.24.2         IRanges_2.22.2             
 [9] S4Vectors_0.26.1            BiocGenerics_0.34.0        
[11] jsonlite_1.7.1              readxl_1.3.1               
[13] crayon_1.3.4                testthat_3.0.0             
[15] margins_0.3.23              mfx_1.2-2                  
[17] betareg_3.1-4               MASS_7.3-53                
[19] lmtest_0.9-38               zoo_1.8-8                  
[21] sandwich_3.0-0              survey_4.0                 
[23] Matrix_1.2-18               Hmisc_4.4-1                
[25] ggplot2_3.3.2               Formula_1.2-4              
[27] survival_3.2-7              lattice_0.20-41            
[29] car_3.0-10                  carData_3.0-4              
[31] foreach_1.5.1               dplyr_1.0.2                
[33] broom_0.7.2                 hash_2.2.6.1               
[35] IHW_1.16.0                 

loaded via a namespace (and not attached):
 [1] colorspace_2.0-0       ellipsis_0.3.1         modeltools_0.2-23     
 [4] rio_0.5.16             htmlTable_2.1.0        XVector_0.28.0        
 [7] base64enc_0.1-3        rstudioapi_0.13        bit64_4.0.5           
[10] flexmix_2.3-17         AnnotationDbi_1.50.3   codetools_0.2-16      
[13] splines_4.0.3          geneplotter_1.66.0     knitr_1.30            
[16] annotate_1.66.0        cluster_2.1.0          png_0.1-7             
[19] compiler_4.0.3         backports_1.2.0        htmltools_0.5.0       
[22] tools_4.0.3            gtable_0.3.0           glue_1.4.2            
[25] GenomeInfoDbData_1.2.3 Rcpp_1.0.5             slam_0.1-47           
[28] cellranger_1.1.0       vctrs_0.3.5            iterators_1.0.13      
[31] xfun_0.19              stringr_1.4.0          openxlsx_4.2.3        
[34] lifecycle_0.2.0        XML_3.99-0.5           zlibbioc_1.34.0       
[37] scales_1.1.1           hms_0.5.3              RColorBrewer_1.1-2    
[40] curl_4.3               memoise_1.1.0          gridExtra_2.3         
[43] rpart_4.1-15           latticeExtra_0.6-29    stringi_1.5.3         
[46] RSQLite_2.2.1          genefilter_1.70.0      checkmate_2.0.0       
[49] zip_2.1.1              BiocParallel_1.22.0    rlang_0.4.8           
[52] pkgconfig_2.0.3        bitops_1.0-6           lpsymphony_1.16.0     
[55] purrr_0.3.4            prediction_0.3.14      htmlwidgets_1.5.2     
[58] bit_4.0.4              tidyselect_1.1.0       magrittr_2.0.1        
[61] R6_2.5.0               generics_0.1.0         DBI_1.1.0             
[64] pillar_1.4.6           haven_2.3.1            foreign_0.8-80        
[67] withr_2.3.0            abind_1.4-5            RCurl_1.98-1.2        
[70] nnet_7.3-14            tibble_3.0.4           fdrtool_1.2.15        
[73] jpeg_0.1-8.1           locfit_1.5-9.4         data.table_1.13.2     
[76] blob_1.2.1             forcats_0.5.0          digest_0.6.27         
[79] xtable_1.8-4           tidyr_1.1.2            munsell_0.5.0         
[82] mitools_2.4     
```    

Typical install time on a normal desktop computer for all Python and R packages is less than 10 minutes.


</p>
