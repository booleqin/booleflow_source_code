---
title: Eclipse下载安装配置MAC版
top: false
cover: false
mathjax: false
date: 2020-11-14 20:34:28
summary: 写给Eclipse新手的下载－安装－配置详细教程
tags:
  - Eclipse
  - 教程
categories: 计算机技术
---

**写给Eclipse新手的下载－安装－配置详细教程**

## 下载Eclipse

mac电脑是自带javaJDK的，所以可以先尝试直接下载Eclipse安装

建议去官网下载，直接百度Eclipse download，网址：http://www.eclipse.org/downloads/

点击左下角Download Packages

![](download_packages.jpeg)

找到Eclipse IDE for Java EE Developers

![](java_ee_developers.jpeg)

除了Java EE 外，还有Eclipse IDE for Java Developers

![](java_developers.jpeg)

具体下载哪一个可依据情况而定，其中Java是带有用户界面的基本IDE，而Java EE是企业版，功能更多一些，两者比较如下：

![](Java_ee_and_java.jpeg)

点击以后直接进行下载，我这里选择的是Eclipse IDE for Java EE Developers

![](download.jpeg)

解压后直接点击Eclipse.app进行安装

![](eclipse_app.jpeg)

至此下载已经安成，后面就需要看电脑情况进行一些插件配置

## 安装

由于电脑配置不同，需要单独下载的插件不同，我这里起始状态为电脑默认配置

双击Eclipse产生无法继续安装的情况，当时没有截图，大概意思是需要下载一个OS X插件，系统提示信息中提供了一个链接，点开后如下

![](download_java.jpeg)

直接下载，是.dmg文件，可直接安装，持续下一步即可

安装成功后重新双击Eclipse，发现上面的问题没有了，接下来出现一个新的问题。截图类似如下（当时忘记截图，后来找的），截止目前最新的是1.8版本

![](jvm.jpeg)

说明系统默认的1.6版JDK不满足要求，需要更新JDK

## 更新JDK有两种方式

**第一种（推荐）**

去oracle官网下载，网址：http://www.oracle.com/technetwork/java/javase/downloads/index.html

![](oracle.jpeg)

现在最新版本已到Kit 8

![](kit8.jpeg)

下载后为.dmg文件，可直接安装，持续下一步即可

默认安装路径为：/Library/Java/JavaVirtualMachines/

![](jdkjdk.jpeg)

安装完成后就需要配置环境变量了，虽然安装了1.8，但是系统默认JDK还是1.6，所以现在还是不能运行的可以打开终端输入：Java –version查看

![](Java_version.jpeg)

此时输入：open -t ~/.bash_profile

在弹出文本中添加

export JAVA_8_HOME=/Library/Java/JavaVirtualMachines/jdk1.8.0_131.jdk/Contents/Home 并 export JAVA_HOME=$JAVA_8_HOME

保存退出重启终端后就已经是1.8了

**第二种更新JDK方式**

系统自动更新

![](gengxin.jpeg)

更新路径可以在这里查看

![](lujing.jpeg)

按照上面的方式配置环境变量即可。

安装完成后打开第一个页面是欢迎页面，关闭即可

![](welcome.jpeg)

在正式写代码之前还要进行些必要的配置

## 打开Eclipse-偏好设置

![](pianhao.jpeg)

此时需要看两个路径是否一至，如果一致可不用管

![](check_path.jpeg)

如果不一致，需要新增一个路径

![](add_path.jpeg)

![](add_jre.jpeg)

![](jre2.jpeg)

![](jre3.jpeg)

现在就已经能正常工作了，除此之外，还可以进行一些其他配置

## 其他配置

**编码格式设置**

![](bianmageshi.jpeg)

**拼写检查设置**

![](pinxiejiancha.jpeg)

**主题背景设置**

![](theme.jpeg)

程序主题设置，这里需要安装一个theme插件

有两种方式

第一种是通过Marketplace安装

![](marketplace.jpeg)

搜索color eclipse theme

![](eclipse_theme.jpeg)

此警告说明没有数字证书，可忽略

![](security.jpeg)

然后重启Eclipse

第二种是通过install入口

![](install.jpeg)

复制链接：http://eclipse-color-theme.github.com/update

Name 可以任取

![](name_1.jpeg)

![](name_2.jpeg)

![](name_3.jpeg)

到这里就已经初步配置完成了

现在开始HelloWorld

## 第一个程序－HelloWorld

新建一个Java Project

![](new_project.jpeg)

![](new_project2.jpeg)

新建一个class

![](new_class1.jpeg)

![](new_class2.jpeg)

打印HelloWorld

![](helloworld.jpeg)



完～

