# KongYiji.jl
断文识字的“孔乙己” -- 一个简单的中文分词工具
Kong Yiji, a simple fine tuned Chinese tokenizer

## Features

### Version 0.1.0
                
1. Trained on Chinese Treebank 8.0. Of version 1 now, using a extended word-level Hidden Markov Model(HMM) contrast by eariler char-level HMM. 

2. Fine tuned to deal with **new words**(未登录词, 网络新词). If the algorithm cannot find them, just add them to user dict(see **Constructor**), and twist **usr_dict_weight** if necessary.

3. Fully exported debug info with functions below:
	1. **postable** : table of part-of-speech(pos) tags used in CTB
	2. **h2vtable** : table of hidden to visual, i.e., table of pos tags to words
	3. **v2htable** : table of hidden to visual, i.e., table of pos tags to words 

## Constructor
```julia
kong(; user_dict_path="", user_dict_array=[], user_dict_weight=1)
```
        
+  **user_dict_path** : a file path of user dict, eachline of which begin a **word**, optionally followed by a **part-of-speech tag(postag)**;
                               If the postag not supplied, **NR (Proper noun, 专有名词)** is automatically inserted. 
+ **user_dict_array** : a Vector{Tuple{String, String}} repr. [(postag, word)]
        
+ **user_dict_weight** : if value is **m**, frequency of (postag, word) in user dictionary will be $ m * maximum(values(h2v[postag])) $

***Note all user suppiled postags MUST conform to specifications of Chinese Treebank.***

## Usage
see test/runtests.jl

## Todos
+ Filter low frequency words from CTB
+ Exploit summary of POS table, insert a example column, plus constract with other POS scheme(PKU etc.)
+ Explore MaxEntropy & CRF related algorithms
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIyNjIyODk5Myw5NjQ3MTk5NDgsODE5MT
U3MzM5LDE0MjI3MDQ2ODYsLTEyNDI5Nzk3MTUsLTIwMDY4ODQ4
NF19
-->