---
title: neo4j使用详解
top: false
cover: false
mathjax: false
date: 2021-03-13 11:11:19
summary: Neo4j是一个基于JAVA编写的NoSQL数据库，相比于 MySQL 之类的关系数据库(RDBMS)，能更灵活地表示数据。
tags:
  - neo4j
  - 数据库
  - Cypher
categories: 计算机技术
---

## 安装neo4j

### neo4j简介

Neo4j是一个基于JAVA编写的NoSQL数据库，相比于 MySQL 之类的关系数据库(RDBMS)，能更灵活地表示数据，主要特点有：

- 可以灵活地设计、扩展 schema
- 适合表示实体之间的关系（特别是当实体之间存在大量、复杂的关系时）

Neo4j存储的图由顶点（节点node）、边（关系relationship）和属性（property）组成，顶点和边都可以设置属性（一个或多个），使用Cypher进行查询，同时支持scala、python等调用。

**开发者可参考：[nep4j developer](https://neo4j.com/developer/)**

### 准备java环境

（如果机器有java环境，此步略）

在shell运行如下命令查看java版本

```shell
java -version
```

Neo4j基于Java环境，所以在安装neo4j之前先安装JAVA SE的JRE，去Oracle官网下载JDK（JDK包含JRE，JRE提供环境，JDK可以支持开发Java程序）。[下载JAVA SDK](https://www.oracle.com/java/technologies/javase-downloads.html) 

- JAVA版本需要与Neo4j版本对应，我使用的neo4j版本4.2.2，使用的JAVA版本jdk-11.0.9

可以把JAVA JDK配置在环境变量也可以在运行时通过export指定（后面使用export指定的方法）。

### 安装neo4j

安装过程非常方便，下载tar包解压，按需修改config即可

下载neo4j，[neo4j下载地址](https://neo4j.com/download-center/#community) ,社区版本是免费的，推荐。

下载完成后解压

```shell
tar -axvf neo4j-community-4.2.2-unix.tar.gz
```

### 修改相应配置

配置在conf/neo4j.conf，并不是全部都需要修改，看个人需要，我就改了一下端口（注：neo4j涉及三个端口），重点关注配置

```shell
# 修改第22行load csv时l路径，在前面加个#，可从任意路径读取文件
#dbms.directories.import=import

# 修改35行和36行，设置JVM初始堆内存和JVM最大堆内存
# 生产环境给的JVM最大堆内存越大越好，但是要小于机器的物理内存
dbms.memory.heap.initial_size=5g
dbms.memory.heap.max_size=10g

# 修改46行，可以认为这个是缓存，如果机器配置高，这个越大越好
dbms.memory.pagecache.size=10g

# 修改54行，去掉改行的#，可以远程通过ip访问neo4j数据库
dbms.connectors.default_listen_address=0.0.0.0

# 默认 bolt端口是7687，http端口是7474，https端口是7473，不修改下面3项也可以
# 修改71行，去掉#，设置http端口为7687，端口可以自定义，只要不和其他端口冲突就行
dbms.connector.bolt.listen_address=:7687

# 修改75行，去掉#，设置http端口为7474，端口可以自定义，只要不和其他端口冲突就行
dbms.connector.http.listen_address=:7474

# 修改79行，去掉#，设置http端口为7473，端口可以自定义，只要不和其他端口冲突就行
dbms.connector.https.listen_address=:7473

# 修改227行，去掉#，允许从远程url来load csv
dbms.security.allow_csv_import_from_file_urls=true

# 修改246行，允许使用neo4j-shell，类似于mysql 命令行之类的
dbms.shell.enabled=true

# 修改235行，去掉#，设置连接neo4j-shell的端口，一般都是localhost或者127.0.0.1，这样安全，其他地址的话，一般使用https就行
dbms.shell.host=127.0.0.1

# 修改250行，去掉#，设置neo4j-shell端口，端口可以自定义，只要不和其他端口冲突就行
dbms.shell.port=1337

# 修改254行，设置neo4j可读可写
dbms.read_only=false

```

### 启动

如果将JAVA配置在环境变量中，则可以直接运行start启动

```shell
bin/neo4j start  # 启动
bin/neo4j stop  # 停止
bin/neo4j restart  # 重启
bin/neo4j status  # 状态
```

如果JAVA没有配置在环境变量可以新建一个执行脚本，类似如下

```shell
export JAVA_HOME="/xxx/jdk-11.0.9"
export PATH=/xxx/neo4j-community-4.2.2/bin:/xxx/jdk-11.0.9/bin:$PATH

neo4j $*
```

### 设置密码

安装完Neo4j后 默认的初始用户名是:neo4j ，密码也是：neo4j

登录成功后，会要求重置 neo4j 密码，修改并确认新密码。也可以通过命令行修改密码，运行

```shell
./cypher-shell  # 命令， 输入用户名，密码（初始默认用户名和密码为 neo4j）
# 输入命令
CALL dbms.security.changePassword('123456'); 
# 输入命令
exit;  # 退出
```

## 数据导入

### 数据导入方法

neo4j提供create、load、insert、import、Neo4j-import 共计5总导入方法。

常见导入形式对比

|          | CREATE语句             | LOAD CSV语句                                 | Batch Inserter                                       | Batch Import                                                 | Neo4j-import                                                 |
| :------- | :--------------------- | :------------------------------------------- | :--------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 适用场景 | 1 ~ 1w nodes           | 1w ~ 10 w nodes                              | 千万以上 nodes                                       | 千万以上 nodes                                               | 千万以上 nodes                                               |
| 速度     | 很慢 (1000 nodes/s)    | 一般 (5000 nodes/s)                          | 非常快 (数万 nodes/s)                                | 非常快 (数万 nodes/s)                                        | 非常快 (数万 nodes/s)                                        |
| 优点     | 使用方便，可实时插入。 | 使用方便，可以加载本地/远程CSV；可实时插入。 | 速度相比于前两个，有数量级的提升                     | 基于Batch Inserter，可以直接运行编译好的jar包；可以在已存在的数据库中导入数据 | 官方出品，比Batch Import占用更少的资源                       |
| 缺点     | 速度慢                 | 需要将数据转换成CSV                          | 需要转成CSV；只能在JAVA中使用；且插入时必须停止neo4j | 需要转成CSV；必须停止neo4j                                   | 需要转成CSV；必须停止neo4j；只能生成新的数据库，而不能在已存在的数据库中插入数据。 |

此处给出Neo4j-import和load两种方法

### Neo4j-import导入

这里默认库式neo4j，所以在使用Neo4j-import之前需要将neo4j-community-4.2.2/data/databases/neo4j 下的文件清空。

**停止neo4j（neo4j stop）**

需要准备将数据准备为CSV格式，实体Node数据格式如下：

![](node_ip_csv.png)

关系数据格式如下

![](edge_ip_uid_csv.png)

导入详细配置如下

```shell
neo4j-admin import [--mode=csv] [--database=<name>]
                          [--additional-config=<config-file-path>]
                          [--report-file=<filename>]
                          [--nodes[:Label1:Label2]=<"file1,file2,...">]
                          [--relationships[:RELATIONSHIP_TYPE]=<"file1,file2,...">]
                          [--id-type=<STRING|INTEGER|ACTUAL>]
                          [--input-encoding=<character-set>]
                          [--ignore-extra-columns[=<true|false>]]
                          [--ignore-duplicate-nodes[=<true|false>]]
                          [--ignore-missing-nodes[=<true|false>]]
                          [--multiline-fields[=<true|false>]]
                          [--delimiter=<delimiter-character>]
                          [--array-delimiter=<array-delimiter-character>]
                          [--quote=<quotation-character>]
                          [--max-memory=<max-memory-that-importer-can-use>]
                          [--f=<File containing all arguments to this import>]
                          [--high-io=<true/false>]
```

我使用用的命令如下

```shell
export JAVA_HOME="/xxx/jdk-11.0.9"
export PATH=/xxx/neo4j-community-4.2.2/bin:/xxx/jdk-11.0.9/bin:$PATH

neo4j-admin import --database=neo4j \
             --nodes ./data/node_ip.csv \
             --nodes ./data/node_mobile.csv \
             --nodes ./data/node_userid.csv \
             --relationships ./data/edge_mobile_userid.csv \
             --relationships ./data/edge_ip_mobile.csv \
             --relationships ./data/edge_ip_userid.csv \
             --skip-duplicate-nodes=true \
             --ignore-empty-strings=true
```

**启动neo4j（neo4j start）即可。**

### load csv 导入

对于数据量千万以下，个人推荐使用load csv 方法，将数据转化为csv后，不用停止neo4j，灵活方便。

**这里使用图的时候有一个需求：只查询近一段时间的关系数据，更早时间的不需要，采用的方法是例行化产生小时级划分的csv文件夹，按照load csv的方法，load指定文件夹的数据到neo4j中，更早时间的数据可以删除**

创建节点

```shell
# 创建NodeIp node数据表
CREATE CONSTRAINT ON (i:NodeIp) ASSERT i.id IS UNIQUE;
LOAD CSV WITH HEADERS FROM 'file:///ppp/20210301/00/node_ip.csv' AS line FIELDTERMINATOR ','  
MERGE (:NodeIp { id:line.id ,name: line.name, lab: line.label});

# 创建NodeUserid node数据表
CREATE CONSTRAINT ON (i:NodeUserid) ASSERT i.id IS UNIQUE;
LOAD CSV WITH HEADERS FROM 'file:///ppp/20210301/00/node_userid.csv' AS line FIELDTERMINATOR ','  
MERGE (:NodeUserid { id:line.id ,name: line.name, lab: line.label});
```

创建边（neo4j的关系是任意的，所以方向可忽略）

```shell
LOAD CSV WITH HEADERS FROM "file:///ppp/20210301/00/edge_ip_userid.csv" AS line FIELDTERMINATOR ','
MATCH (from:NodeIp{id: line.startid}),(to:NodeUserid{id: line.endid}) MERGE (from)-[r:ip_userid {typ: line.typ, appname:line.appname, dtime:line.dtime, dt:line.dt, hour:line.hour}]->(to);
```

创建完成后如下

![](ip_userid_graph.png)

## Cypher操作

### CQL语言

CQL代表Cypher查询语言。 像Oracle数据库具有查询语言SQL，Neo4j具有CQL作为查询语言。详细可参考 [neo4j教程](https://www.w3cschool.cn/neo4j/neo4j_cql_return_clause.html)

在创建节点和边的过程中已经使用过CREATE、MATCH、MERGE、LAOD等语句。

```shell
# NodeIp节点表，并显示前25行。
MATCH (n:NodeIp) RETURN n LIMIT 25;
```

```shell
# 加where条件
MATCH (IP:NodeIp) 
WHERE IP.name = 'xxx.xxx.xxx.xxx'
RETURN IP LIMIT 25;
```

```shell
# 删除节点
MATCH (IP:NodeIp) DELETE IP;
```

```shell
# 添加或更新属性值
MATCH (IP:NodeIp)
SET IP.name = 'xxx'
RETURN IP;
```

```shell
# 排序
MATCH (IP:NodeIp)
RETURN IP.name
ORDER BY IP.name DESC;
```

```shell
# in 运算符
MATCH (IP:NodeIp) 
WHERE IP.name IN ['xxx','xxx']
RETURN IP;
```

更多可参考 [neo4j教程](https://www.w3cschool.cn/neo4j/neo4j_cql_return_clause.html)



完～

