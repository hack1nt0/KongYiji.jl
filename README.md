# KongYiji.jl
断文识字的‘’孔乙己‘’ -- 一个简单的中文分词工具
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

### Note all user suppiled postags MUST conform to specifications of Chinest Treebank.

## Usage

``` Julia
tk = Kong()
input = "一个脱离了低级趣味的人"
output = tk(input)
@show output

input = "一/个/脱离/了/低级/趣味/的/人"
tk(input, "/")
inputs = [
        "他/说/的/确实/在理",
        "这/事/的确/定/不/下来",
        "费孝通/向/人大/常委会/提交/书面/报告",
        "邓颖超/生前/使用/过/的/物品",
        "停电/范围/包括/沙坪坝区/的/犀牛屙屎/和/犀牛屙屎抽水",
]
        println("Input :")
        for input in inputs
                println(input)
        end

        println("raw output :")
        for input in inputs
                println(tk(filter(c -> c != '/', input)))
        end
        
        tk2 = Kong(; user_dict_array=[("VV", "定"),
                                      ("VA", "在理"),
                                       "邓颖超",
                                       "沙坪坝区", 
                                       "犀牛屙屎",
                                       "犀牛屙屎抽水",
                                     ]
        )
        println("output with user dict supplied :")
        for input in inputs
                println(tk2(filter(c -> c != '/', input)))
        end
```

## Output
```
output = ["一", "个", "脱离", "了", "低级", "趣味", "的", "人"]
Standard : 一  个  脱离  了  低级  趣味  的  人
Output   : 一  个  脱离  了  低级  趣味  的  人
          KongYiji(1) Debug Table
  -----------------------------------------
     word pos.tag source prob.h2v Prob.Add.
  1    一      CD    CTB 0.323977  0.203435
  2    个       M    CTB 0.260022  0.019667
  3  脱离      VV    CTB 0.000177    1.1e-5
  4    了      AS    CTB 0.808087  0.045661
  5  低级      JJ    CTB 0.000462  0.000352
  6  趣味      NN    CTB   4.2e-5    2.0e-6
  7    的     DEG    CTB 0.972857  0.744126
  8    人      NN    CTB  0.01615  0.004388
  =========================================
  neg.log.likelihood = 50.088239033558935

  AhoCorasickAutomaton Matched Words
  ---------------------------
      UInt8.range word source
  1        (1, 4)   一    CTB
  2        (4, 7)   个    CTB
  3       (7, 10)   脱    CTB
  4       (7, 13) 脱离    CTB
  5      (10, 13)   离    CTB
  6      (13, 16)   了    CTB
  7      (16, 19)   低    CTB
  8      (16, 22) 低级    CTB
  9      (19, 22)   级    CTB
  10     (22, 25)   趣    CTB
  11     (22, 28) 趣味    CTB
  12     (25, 28)   味    CTB
  13     (28, 31)   的    CTB
  14     (31, 34)   人    CTB

Input :
他/说/的/确实/在理
这/事/的确/定/不/下来
费孝通/向/人大/常委会/提交/书面/报告
邓颖超/生前/使用/过/的/物品
停电/范围/包括/沙坪坝区/的/犀牛屙屎/和/犀牛屙屎抽水
raw output :
["他", "说", "的", "确实", "在", "理"]
["这", "事", "的", "确定", "不", "下来"]
["费孝通", "向", "人大", "常委会", "提交", "书面", "报告"]
["邓", "颖", "超生", "前", "使用", "过", "的", "物品"]
["停电", "范围", "包括", "沙", "坪", "坝", "区", "的", "犀牛", "屙", "屎", "和", "犀牛", "屙", "屎", "抽水"]
output with user dict supplied :
["他", "说", "的", "确实", "在理"]
["这", "事", "的确", "定", "不", "下来"]
["费孝通", "向", "人大", "常委会", "提交", "书面", "报告"]
["邓颖超", "生前", "使用", "过", "的", "物品"]
["停电", "范围", "包括", "沙坪坝区", "的", "犀牛屙屎", "和", "犀牛屙屎抽水"]
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTYzMjczMzAzM119
-->