---
title: linux下crontab服务的使用
tags:
---

## crontab 命令

crontab是用来定期执行程序的命令

## crontab服务的启动停止

### 使用service命令（推荐）

一般情况下，linux自带有service命令，可以启动与停止crontab服务

```shell
# 启动
service crond start
# 停止
service crond stop
# 重启
service crond restart
# 重载
service cron reload
```

### 不使用service命令

当linux发行的版本没有service这个命令时，可使用如下命令启动。

```shell
# 启动
/etc/init.d/cron start
# 停止
/etc/init.d/cron stop
```

## 管理crontab命令

```shell
crontab -e  # 执行文字编辑器来设定时程表
crontab -r  # 删除目前的时程表
crontab -l  # 列出目前的时程表
crontab file [-u user]  # 用指定的文件替代目前的crontab。
```

