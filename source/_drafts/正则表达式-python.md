---
title: 正则表达式+python
top: false
cover: false
mathjax: false
date: 2021-09-03 17:15:21
summary:
tags:
categories:
---



## 1.今天在做一道匹配邮箱的题目，发现re.findall()匹配的时候如果有括号，只能取到括号里面的内容：

```text
有以下字符串y，使用正则表达式匹配出字符串中的邮箱地址。
y= '123@qq.comaaa@163.combbb@126.comasdfasfs33333@adfcom'
```

要是想匹配邮箱地址的话，必须做一些其他的工作，下面提供了三种写法：

```python
import re
y= '123@qq.comaaa@163.combbb@126.comasdfasfs33333@adfcom'
#第一种写法：
ret1=re.findall('[0-9a-z]+@+[0-9a-z]+\.com',y)
print(ret1)
##第二种写法：
ret2=re.findall('(\w+@(qq|126|163)\.com)',y)
for i in ret2:
    print(i[0])
#第三种写法：
ret3=re.findall('\w+@(?:qq|126|163)\.com',y)
print(ret3)
```

参考：[python 正则表达式findall匹配问题](https://zhuanlan.zhihu.com/p/37900841)

## 换行

```text
re.search(r'aaa'
          r'bbb'
          r'ccc','aaabbbccc')
```

或

```python
surname += r"|后|况|亢|缑|帅|微生|羊舌|海|归|呼延|南门|东郭|百里|钦|鄢|汝|法|闫|楚|晋|谷梁|宰父|夹谷|拓跋|壤驷|乐正|漆雕|公西|巫马|端木|颛孙"
surname += r"|子车|督|仉|司寇|亓官|三小|鲜于|锺离|盖|逯|库|郏|逢|阴|薄|厉|稽|闾丘|公良|段干|开|光|操|瑞|眭|泥|运|摩|伟|铁|迮]"
surname += r"[\u4e00-\u9fa5]{1,3})(?:[^\u4e00-\u9fa5])"
```

参考：[Python正则表达式过长时如何换行？](https://www.zhihu.com/question/40865347)

