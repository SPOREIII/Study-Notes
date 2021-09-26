# Python相关代码

# 时间序列

- 生成时间序列index

  ```python
  import pandas as pd
  time_index = pd.date_range(start='2019-06-20 06:00:00', 
                             end='2019-06-22 15:00:00', freq='H')# freq参数可以指定生成序列的时间间隔'H'每小时，'D'每天等等
  print(time_index)
  ```

  ```bash
  DatetimeIndex(['2019-06-20 06:00:00', '2019-06-20 07:00:00',
                 '2019-06-20 08:00:00', '2019-06-20 09:00:00',
                 '2019-06-20 10:00:00', '2019-06-20 11:00:00',
                 '2019-06-20 12:00:00', '2019-06-20 13:00:00',
                 '2019-06-20 14:00:00', '2019-06-20 15:00:00',
                 '2019-06-20 16:00:00', '2019-06-20 17:00:00',
                 '2019-06-20 18:00:00', '2019-06-20 19:00:00',
                 '2019-06-20 20:00:00', '2019-06-20 21:00:00',
                 '2019-06-20 22:00:00', '2019-06-20 23:00:00',
                 '2019-06-21 00:00:00', '2019-06-21 01:00:00',
                 '2019-06-21 02:00:00', '2019-06-21 03:00:00',
                 '2019-06-21 04:00:00', '2019-06-21 05:00:00',
                 '2019-06-21 06:00:00', '2019-06-21 07:00:00',
                 '2019-06-21 08:00:00', '2019-06-21 09:00:00',
                 '2019-06-21 10:00:00', '2019-06-21 11:00:00',
                 '2019-06-21 12:00:00', '2019-06-21 13:00:00',
                 '2019-06-21 14:00:00', '2019-06-21 15:00:00',
                 '2019-06-21 16:00:00', '2019-06-21 17:00:00',
                 '2019-06-21 18:00:00', '2019-06-21 19:00:00',
                 '2019-06-21 20:00:00', '2019-06-21 21:00:00',
                 '2019-06-21 22:00:00', '2019-06-21 23:00:00',
                 '2019-06-22 00:00:00', '2019-06-22 01:00:00',
                 '2019-06-22 02:00:00', '2019-06-22 03:00:00',
                 '2019-06-22 04:00:00', '2019-06-22 05:00:00',
                 '2019-06-22 06:00:00', '2019-06-22 07:00:00',
                 '2019-06-22 08:00:00', '2019-06-22 09:00:00',
                 '2019-06-22 10:00:00', '2019-06-22 11:00:00',
                 '2019-06-22 12:00:00', '2019-06-22 13:00:00',
                 '2019-06-22 14:00:00', '2019-06-22 15:00:00'],
                dtype='datetime64[ns]', freq='H')
  ```


- `str`格式转`datetime`格式

  ```python
  import pandas as pd
  time = ['2018-10-31 03:00:00', '2018-10-31 04:00:00', '2018-10-31 05:00:00', '2018-10-31 06:00:00']
  time = pd.to_datetime(time)
  print(time)
  ```

  ```python
  DatetimeIndex(['2018-10-31 03:00:00', '2018-10-31 04:00:00',
                 '2018-10-31 05:00:00', '2018-10-31 06:00:00'],
                dtype='datetime64[ns]', freq=None)
  ```

- `datetime`转`str`格式

  ```python
  import pandas as pd
  time = ['2018-10-31 03:00:00', '2018-10-31 04:00:00', '2018-10-31 05:00:00', '2018-10-31 06:00:00']
  time = pd.to_datetime(time)
  print(type(time[0]))
  time = time.strftime("%Y-%m-%d %H:%M:%S")
  print(type(time[0]))
  print(time)
  ```
  
  ```bash
  <class 'pandas._libs.tslibs.timestamps.Timestamp'>
  <class 'str'>
  Index(['2018-10-31 03:00:00', '2018-10-31 04:00:00', '2018-10-31 05:00:00',
         '2018-10-31 06:00:00'],
        dtype='object')
  ```
  
- 增加或减少时间

  ```python
  import datetime
  time = datetime.datetime.now()
  time_plus_one_hour = time+datetime.timedelta(hours=1)
  print(time)
  print(time_plus_one_hour)
  ```

  ```bash
  2021-08-12 15:37:24.183138
  2021-08-12 16:37:24.183138
  ```


# numpy

- 合并两数组

  ```python
  import numpy as np
  a = np.array([[1,2,3]])
  b = np.array([[1,2,3]])
  result = np.concatenate([a,b], axis=0)
  print(result)
  ```

  ```bash
  [[1 2 3]
   [1 2 3]]
  ```

- 提取某些列

  ```python
  a = np.array([[1,2,3],[4,5,6]])
  a[:,[0,2]]
  ```

  ```bash
  array([[1, 3],
         [4, 6]])
  ```

- 数学方法

  ```python
  import numpy as np
  np.square() # 平方
  np.sqrt() # 平方根
  np.maximum() # 从两个输入中选大
  np.max() # 从序列输入中选大
  ```

# Pandas

- 多个dataframe写入同一个`.xlsx`的不同sheet：

  - 写入

    ```python
    import pandas as pd
    a = pd.DataFrame({'a':[1,2,3],'b':[4,5,6]})
    y = pd.DataFrame({'x':[11,22,33],'y':[44,55,66]})
    writer = pd.ExcelWriter('test.xlsx')
    a.to_excel(writer, 'a')
    y.to_excel(writer, 'y')
    writer.save()
    ```

  - 读取

    ```python
    c = pd.read_excel('test.xlsx', sheet_name='a', index_col=0)
    d = pd.read_excel('test.xlsx', sheet_name='y', index_col=0)
    print(c)
    print(d)
    ```

    ```bash
       a  b
    0  1  4
    1  2  5
    2  3  6
        x   y
    0  11  44
    1  22  55
    2  33  66
    ```

- 读取数据并绘制可交互的图标

  spyder编辑器中，`工具>偏好>IPython控制台>绘图>图形的后端>选择(自动)`

  ```python
  import pandas as pd
  dataset = pd.read_csv('./dataset/dataset.csv', index_col='time')
  dataset.index = pd.to_datetime(dataset.index)
  dataset = dataset[['3721-W', '3722-W', '5921-W', '5173-W']]
  dataset.plot()
  ```

- 设置双`index`

  ```python
  import pandas as pd
  flood_event = ['19-06-19', '19-07-09', '19-05-26', '19-06-30', '19-02-12']
  aim_hourly = [6, 12]
  site = ['3721-W','3722-W','5921-W','5173-W']
  indicators = ['q10', 'q50', 'q90', 'NSE', 'QT', 'QL']
  table = np.zeros((len(flood_event)*len(aim_hourly), len(site)+2))
  table = pd.DataFrame(table, columns=['flood', 'horizon', *site])
  table['horizon'] = [i for j in flood_event for i in aim_hourly]
  table['flood'] = [j for j in flood_event for i in aim_hourly]
  # =====================构建dataframe======================================
  table.set_index(['flood', 'horizon'], inplace=True) #设置双index，inplace参数为true时直接修改原表，或者可写下面形式
  # table = table.set_index(['flood', 'horizon'])
  print(table)
  print(table.loc['19-06-19',6])
  table['3722-W'].loc['19-06-19',6] = 6
  print(table)
  ```

  ```bash
                    3721-W  3722-W  5921-W  5173-W
  flood    horizon                                
  19-06-19 6           0.0     0.0     0.0     0.0
           12          0.0     0.0     0.0     0.0
  19-07-09 6           0.0     0.0     0.0     0.0
           12          0.0     0.0     0.0     0.0
  19-05-26 6           0.0     0.0     0.0     0.0
           12          0.0     0.0     0.0     0.0
  19-06-30 6           0.0     0.0     0.0     0.0
           12          0.0     0.0     0.0     0.0
  19-02-12 6           0.0     0.0     0.0     0.0
           12          0.0     0.0     0.0     0.0
  3721-W    0.0
  3722-W    0.0
  5921-W    0.0
  5173-W    0.0
  Name: (19-06-19, 6), dtype: float64
                    3721-W  3722-W  5921-W  5173-W
  flood    horizon                                
  19-06-19 6           0.0     6.0     0.0     0.0
           12          0.0     0.0     0.0     0.0
  19-07-09 6           0.0     0.0     0.0     0.0
           12          0.0     0.0     0.0     0.0
  19-05-26 6           0.0     0.0     0.0     0.0
           12          0.0     0.0     0.0     0.0
  19-06-30 6           0.0     0.0     0.0     0.0
           12          0.0     0.0     0.0     0.0
  19-02-12 6           0.0     0.0     0.0     0.0
           12          0.0     0.0     0.0     0.0
  ```

# 内置方法

- `list`的解包

  ```python
  a = [1,2,3,4]
  b = [1,2,*a]
  print(b)
  ```

  ```bash
  [1, 2, 1, 2, 3, 4]
  ```

- 保留$n$位小数

  ```python
  a = 0.21315215
  print(round(a, 4))
  ```

  ```bash
  0.2132
  ```

# 深拷贝与浅拷贝

- `list`相关

  ```python
  a = [1, 2, 1, 2, 3, 4]
  b = [a for _ in range(3)]
  print(id(a)) # id()函数返回对象在内存中的地址
  print(id(b[0]))
  print(id(b[1]))
  print(id(b[2]))
  b[0][1] = 7
  print(b)
  print(a)
  ```

  ```bash
  140205612152192
  140205612152192
  140205612152192
  140205612152192
  [[1, 7, 1, 2, 3, 4], [1, 7, 1, 2, 3, 4], [1, 7, 1, 2, 3, 4]]
  [1, 7, 1, 2, 3, 4]
  ```

  `copy.deepcopy`方法可以将浅拷贝转换为深拷贝

  ```python
  import copy
  a = [1, 2, 1, 2, 3, 4]
  b = [copy.deepcopy(a) for _ in range(3)]
  print(id(a))
  print(id(b[0]))
  print(id(b[1]))
  print(id(b[2]))
  b[0][1] = 7
  print(b)
  print(a)
  ```

  ```bash
  140207530513792
  140207530513920
  140207530514304
  140207530514816
  [[1, 7, 1, 2, 3, 4], [1, 2, 1, 2, 3, 4], [1, 2, 1, 2, 3, 4]]
  [1, 2, 1, 2, 3, 4]
  ```

# 列表相关操作

- 交错填充数字

  希望得到这样一个列表`a = [1,2,1,2,1,2,1,2]`

  ```python
  a = [i for j in range(5) for i in [1,2]]
  print(a)
  ```

  ```python
  [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
  ```

- 填充连续数字

  希望得到这样一个列表`a = [1,1,2,2,3,3]`

  ```python
  a = [j for j in range(1, 4) for i in range(2)]
  print(a)
  ```

  ```python
  [1, 1, 2, 2, 3, 3]
  ```

- 删除元素

  删除`list`中第一个和`val`等值的元素

  ```python
  a = [1,2,34,2,5]
  a.remove(2)
  print(a)
  ```

  ```bash
  [1, 34, 2, 5]
  ```

  删除索引对应的元素

  ```python
  a = [1,2,3,4,5]
  b = a.pop(3)
  print(a)
  print(b)
  ```

  ```bash
  [1, 2, 3, 5]
  4
  ```

# 作图

- linux下图片中文字符乱码

  在系统中安装相应字体

  ```python
  import matplotlib.pyplot as plt
  plt.rcParams['font.sans-serif'] = ['SimHei']
  plt.rcParams['axes.unicode_minus'] = False
  ```

- 设置坐标标签的字体大小

  ```python
  import matplotlib.pyplot as plt
  plt.xlabel(..., fontsize=20)
  ```

# 将字典保存为json

```python
import json
a = ['氢','氦','锂','铍','硼','碳','氮','氧','氟','氖',
     '钠','镁','铝','硅','磷','硫','氯','氩','钾',
     '钙','钪','钛','钒','铬','锰','铁','钴','镍',
     '铜','锌','镓','锗','砷','硒','溴','氪','铷',
     '锶','钇','锆','铌','钼','锝','钌','铑','钯','银','镉','铟','锡','锑',
     '碲','碘','氙','铯','钡','镧','铈','镨','钕','钷','钐','铕','钆','铽',
     '镝','钬','铒','铥','镱','镥','铪','钽','钨','铼','锇','铱','铂','金',
     '汞','铊','铅']
b = ['H','He','Li','Be','B','C','N','O','F','Ne',
     'Na','Mg','Al','Si','P','S','Cl','Ar','K',
     'Ca','Sc','Ti','V','Cr','Mn','Fe','Co',
     'Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y',
     'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I',
     'Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho',
     'Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl',
     'Pb']
zh2en = dict(zip(a,b))
en2zh = dict(zip(b,a))

info_json = json.dumps(zh2en,sort_keys=False, indent=4, separators=(',',':'))
f=open('elements_zh2en.json', 'w')
f.write(info_json)
f.close()
# 保存
f2 = open('elements_en2zh.json', 'r')
info_data = json.load(f2)
# 读取

test = {'A':{'cost':200,'name':'666','element':{'pb':0.2,'al':0.8}},
        'B':{'cost':500,'name':'777','element':{'pb':0.5,'al':0.2}}}
info_json = json.dumps(test,sort_keys=False, indent=4, separators=(',',':'))
f=open('test.json', 'w')
f.write(info_json)
f.close()

f2 = open('test.json', 'r')
info_data = json.load(f2)
```

- 如果保存了空字典

  ```python
  test = {}
  info_json = json.dumps(test,sort_keys=False, indent=4, separators=(',',':'))
  f=open('test.json', 'w')
  f.write(info_json)
  f.close()
  
  f2 = open('test.json', 'r')
  info_data = json.load(f2)
  print(len(info_data))
  ```

  ```bash
  0
  ```

# 字典相关操作

```python
dic = {'a':1,'b':2,'c':3}
print(dic.keys())
print(dic.values())
print(dic.items())
```

```bash
dict_keys(['a', 'b', 'c'])
dict_values([1, 2, 3])
dict_items([('a', 1), ('b', 2), ('c', 3)])
```

```python
dic = {'a':[0,0],'b':[0.2,1],'c':[0,0.2],'d':[0.1,0.2]}
print(sum([1 for i in dic.values() if i[0] > 0 or i[1]>0]))
```

```bash
3
```

- 合并两个字典

  ```pyt
  a = {1:2}
  b = {3:4}
  c = {**a,**b}
  print(c)
  ```

  ```bash
  {1: 2, 3: 4}
  ```

# 集合

集合`set`创建一个无序不重复的元素序列

```python
a = set([1,2,3,4,3,2])
a = {[1,2,3,4,3,2]}
b = set()
```

