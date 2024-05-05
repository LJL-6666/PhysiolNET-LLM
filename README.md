# PhysiolNET-LLM

![License](https://img.shields.io/badge/license-MIT-yellow)
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/apps/detail/lengbaihang1/Elysia) ![Static Badge](https://img.shields.io/badge/license-Apache--2.0-green?label=license)

**[English](readme_english.md)|[中文](README.md)**


## 简介
本项目是生理数据大模型，实现心理健康状态的检查与评估。通过腕表等可穿戴设备测量的人们生理信号的多模态数据，来微调LLM解读数据中分析出的情绪和焦虑测评结果，帮助大众了解自我情绪和认识内在需求。


## 数据集
清华大学心理学系团队发布面向情绪研究的日常动态心理生理记录数据集DAPPER

文章全文可通过如下链接或点击阅读全文获取：https://rdcu.be/cnkg6

数据下载：https://cloud.tsinghua.edu.cn/d/1600eafacd69474c9c0e/

## 数据预处理

由于DAPPER中的数据是按各被试者三种生理信号分别存进excel表中，且每种生理信息数据按照不同时间段分成了多个excel表格，在数据采集时可能存在**异常值**和**缺失值**，且为了能够正确输入神经网络，需要对**数据进行清洗**和**特征提取**，以下是我们数据预处理的步骤图：

![数据预处理.png](https://github.com/LJL-6666/PhysiolNET-LLM/blob/main/img/数据预处理.png)



## 模型预训练

## 特别感谢

<div align="center">

非常感谢书生·浦语团队对项目的支持(￣▽￣)~*

感谢 OpenXLab 对项目部署的算力支持ヾ(^▽^ヾ)

</div>
