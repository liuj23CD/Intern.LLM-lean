# 第6课笔记
## Lagent 是什么
Lagent 是一个轻量级开源智能体框架，旨在让用户可以高效地构建基于大语言模型的智能体。同时它也提供了一些典型工具以增强大语言模型的能力。

Lagent 目前已经支持了包括 AutoGPT、ReAct 等在内的多个经典智能体范式，也支持了如下工具：

Arxiv 搜索
Bing 地图
Google 学术搜索
Google 搜索
交互式 IPython 解释器
IPython 解释器
PPT
Python 解释器

## AgentLego 是什么
AgentLego 是一个提供了多种开源工具 API 的多模态工具包，旨在像是乐高积木一样，让用户可以快速简便地拓展自定义工具，从而组装出自己的智能体。通过 AgentLego 算法库，不仅可以直接使用多种工具，也可以利用这些工具，在相关智能体框架（如 Lagent，Transformers Agent 等）的帮助下，快速构建可以增强大语言模型能力的智能体。

AgentLego 目前提供了如下工具：
![image](https://github.com/liuj23CD/Intern.LLM-lean/assets/132553256/a42618da-e7ef-4fc9-9da9-afcdad8eaf1b)

# 通用能力			
计算器
谷歌搜索
# 语音相关
文本 -> 音频（TTS）
音频 -> 文本（STT）
# 图像处理描述输入图像
识别文本（OCR）
视觉问答（VQA）
人体姿态估计
人脸关键点检测
图像边缘提取（Canny）
深度图生成
生成涂鸦（Scribble）
检测全部目标
检测给定目标
SAM
 分割一切
 分割给定目标
# AIGC
文生图
图像拓展
删除给定对象
替换给定对象
根据指令修改
ControlNet 系列
根据边缘+描述生成
根据深度图+描述生成
根据姿态+描述生成
根据涂鸦+描述生成
ImageBind 系列
音频生成图像
热成像生成图像
音频+图像生成图像
音频+文本生成图像
# 两者的关系
Lagent 是一个智能体框架，而 AgentLego 与大模型智能体并不直接相关，而是作为工具包，在相关智能体的功能支持模块发挥作用。
![image](https://github.com/liuj23CD/Intern.LLM-lean/assets/132553256/05351a57-3499-4c4e-aa94-0908cfc593bf)
