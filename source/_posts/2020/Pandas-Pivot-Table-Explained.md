---
title: Pandas_Pivot_Table_Explained
top: false
cover: false
mathjax: false
date: 2020-11-15 21:30:03
summary: python数据分析利器，类似Excel中的数据透视表。
tags:
  - 数据分析
  - 数据透视
  - 统计学
categories: 计算机技术
---

**--Adapted from http://pbpython.com/pandas-pivot-table-explained.html**

**--If you want more information about Practical Business Python ,please visit http://pbpython.com/**

Posted by [Chris Moffitt](http://pbpython.com/author/chris-moffitt.html) in [articles](http://pbpython.com/category/articles.html) 

## Read in the data

Let’s set up our environment first.

If you want to follow along, you can download the [Excel file](http://pbpython.com/extras/sales-funnel.xlsx).

```python
import pandas as pd 
import numpy as np
```

Read in our sales funnel data into our DataFrame

```python
df = pd.read_excel("../in/sales-funnel.xlsx") 
df.head()
```

![](df.jpeg)

For convenience sake, let’s define the status column as a `category` and set the order we want to view.

This isn’t strictly required but helps us keep the order we want as we work through analyzing the data.

```python
df["Status"] = df["Status"].astype("category") df["Status"].cat.set_categories(["won","pending","presented","declined"],inplace=True)
```

## Pivot the data

As we build up the pivot table, I think it’s easiest to take it one step at a time. Add items and check each step to verify you are getting the results you expect. Don’t be afraid to play with the order and the variables to see what presentation makes the most sense for your needs.

The simplest pivot table must have a dataframe and an `index` . In this case, let’s use the Name as our index_name.index.

```python
pd.pivot_table(df,index=["Name"])
```

![](index_name.jpeg)

You can have multiple indexes as well. In fact, most of the `pivot_table` args can take multiple values via a list.

```python
pd.pivot_table(df,index=["Name","Rep","Manager"])
```

![](index3.jpeg)

This is interesting but not particularly useful. What we probably want to do is look at this by Manager and Rep. It’s easy enough to do by changing the `index` .

```python
pd.pivot_table(df,index=["Manager","Rep"])
```

![](manager_rep.jpeg)

You can see that the pivot table is smart enough to start aggregating the data and summarizing it by grouping the reps with their managers. Now we start to get a glimpse of what a pivot table can do for us.

For this purpose, the Account and Quantity columns aren’t really useful. Let’s remove it by explicitly defining the columns we care about using the `values` field.

```python
pd.pivot_table(df,index=["Manager","Rep"],values=["Price"])
```

![](manager_rep_price.jpeg)

The price column automatically averages the data but we can do a count or a sum. Adding them is simple using `aggfunc` and`np.sum` .

```python
pd.pivot_table(df,index=["Manager","Rep"],values=["Price"],aggfunc=np.sum)
```

![](sum_price.jpeg)

`aggfunc` can take a list of functions. Let’s try a mean using the numpy `mean` function and `len` to get a count.

```python
pd.pivot_table(df,index=["Manager","Rep"],values=["Price"],aggfunc=[np.mean,len])
```

![](mean_len.jpeg)

If we want to see sales broken down by the products, the `columns` variable allows us to define one or more columns.

```python
pd.pivot_table(df,index=["Manager","Rep"],values=["Price"], columns=["Product"],aggfunc=[np.sum])
```

![](product_sum.jpeg)

The NaN’s are a bit distracting. If we want to remove them, we could use `fill_value` to set them to 0.

```python
pd.pivot_table(df,index=["Manager","Rep"],values=["Price"], columns=["Product"],aggfunc=[np.sum],fill_value=0)
```

![](fill_value0.jpeg)

I think it would be useful to add the quantity as well. Add Quantity to the `values` list.

```python
pd.pivot_table(df,index=["Manager","Rep"],values=["Price","Quantity"], columns=["Product"],aggfunc=[np.sum],fill_value=0)
```

![](fill_value_1.jpeg)

What’s interesting is that you can move items to the index to get a different visual representation. Remove Product from the `columns`and add to the `index` .

```python
pd.pivot_table(df,index=["Manager","Rep","Product"], values=["Price","Quantity"],aggfunc=[np.sum],fill_value=0)
```

![](fill_value_pq.jpeg)

For this data set, this representation makes more sense. Now, what if I want to see some totals? `margins=True` does that for us.

```python
pd.pivot_table(df,index=["Manager","Rep","Product"], values=["Price","Quantity"], aggfunc=[np.sum,np.mean],fill_value=0,margins=True)
```

![](pq_sum_mean.jpeg)

Let’s move the analysis up a level and look at our pipeline at the manager level. Notice how the status is ordered based on our earlier category definition.

```python
pd.pivot_table(df,index=["Manager","Status"],values=["Price"], aggfunc=[np.sum],fill_value=0,margins=True)
```

![](margins_true.jpeg)

A really handy feature is the ability to pass a dictionary to the `aggfunc` so you can perform different functions on each of the values you select. This has a side-effect of making the labels a little cleaner.

```shell
pd.pivot_table(df,index=["Manager","Status"],columns=["Product"],values=["Quantity","Price"], aggfunc={"Quantity":len,"Price":np.sum},fill_value=0)
```

![](aggfunc.jpeg)

You can provide a list of aggfunctions to apply to each value too:

```shell
pd.pivot_table(df,index=["Manager","Status"],columns=["Product"],values=["Quantity","Price"], aggfunc={"Quantity":len,"Price":[np.sum,np.mean]},fill_value=0) 
```

![](aggfunc2.jpeg)

It can look daunting to try to pull this all together at one time but as soon as you start playing with the data and slowly add the items, you can get a feel for how it works. My general rule of thumb is that once you use multiple `grouby` you should evaluate whether a pivot table is a useful approach.

## Advanced Pivot Table Filtering

Once you have generated your data, it is in a `DataFrame` so you can filter on it using your standard `DataFrame` functions.

If you want to look at just one manager:

```python
table.query('Manager == ["Debra Henley"]')
```

![](query_manager.jpeg)

We can look at all of our pending and won deals.

```python
table.query('Status == ["pending","won"]')
```

![](query_status.jpeg)

This a poweful feature of the `pivot_table` so do not forget that you have the full power of pandas once you get your data into the`pivot_table` format you need.

The full [notebook](http://nbviewer.ipython.org/url/pbpython.com/extras/Pandas-Pivot-Table-Explained.ipynb) is available if you would like to save it as a reference.

## Cheat Sheet

n order to try to summarize all of this, I have created a cheat sheet that I hope will help you remember how to use the pandas`pivot_table` . Take a look and let me know what you think.

![](cheat_sheet.jpeg)

Thanks and good luck with creating your own pivot tables.

end～