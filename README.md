# KongYiji.jl
断文识字的“孔乙己” -- 一个简单的中文分词工具
Kong Yiji, a simple fine tuned Chinese tokenizer

## Features

### Version 0.1.0
                
1. Trained on Chinese Treebank 8.0. Of version 1 now, using a extended word-level Hidden Markov Model(HMM) contrast by eariler char-level HMM. 

2. Fine tuned to deal with **new words**(未登录词, 网络新词). If the algorithm cannot find them, just add them to user dict(see **Constructor**), and twist **usr_dict_weight** if necessary.

3. Fully exported debug info. See Usage.

## Constructor
```julia
kong(; user_dict_path="", user_dict_array=[], user_dict_weight=1)
```
        
+  **user_dict_path** : a file path of user dict, eachline of which begin a **word**, optionally followed by a **part-of-speech tag(postag)**;
                               If the postag not supplied, **NR (Proper noun, 专有名词)** is automatically inserted. 
+ **user_dict_array** : a Vector{Tuple{String, String}} repr. [(postag, word)]
        
+ **user_dict_weight** : if value is **m**, frequency of (postag, word) in user dictionary will be $ m * maximum(values(h2v[postag])) $

***Note all user suppiled postags MUST conform to specifications of Chinese Treebank.***
```
                                     CTB postable
  -------------------------------------------------------------------------------------
      ﻿part.of.speech                                                            summary
  1               NR                                                           专属名词
  2               NT                                                               时间
  3               NN                                                           其他名词
  4               PN                                                               代词
  5               VA                                                       形容词动词化
  6               VC                                              be、not be 对应的中文
  7               VE                                          have、not have 对应的中文
  8               VV                                                           其他动词
  9                P                                                               介词
  10              LC                                                             方位词
  11              AD                                                               副词
  12              DT                                                       谁的，哪一个
  13              CD                                                         （数）量词
  14              OD                                                         （顺）序词
  15               M                                                         （数）量词
  16              CC                                                         连（接）词
  17              CS                                                         连（接）词
  18             DEC                                                                 的
  19             DEG                                                                 的
  20             DER                                                                 得
  21             DEV                                                                 地
  22              AS Aspect Particle 表达英语中的进行式、完成式的词，比如（着，了，过）
  23              SP                               句子结尾词（了，吧，呢，啊，呀，吗）
  24             ETC                                                           等（等）
  25             MSP                                                               其他
  26              IJ                                                         句首感叹词
  27              ON                                                             象声词
  28              LB                                                                 被
  29              SB                                                                 被
  30              BA                                                                 把
  31              JJ                                                         名词修饰词
  32              PU                                                           标点符号
  33              FW                                        POS不清楚的词（不是外语词）
```

## Usage


## Todos
+ Filter low frequency words from CTB
+ Exploit summary of POS table, insert a example column, plus constract with other POS scheme(PKU etc.)
+ Explore MaxEntropy & CRF related algorithms
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTgwNTYzMDg5NiwxNDIyNzA0Njg2LC0xMj
QyOTc5NzE1LC0yMDA2ODg0ODRdfQ==
-->