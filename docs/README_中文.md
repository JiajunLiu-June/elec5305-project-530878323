# ELEC5305 最终版使用说明

Jiajun Liu，学号 **550378323**。

## 三项提交材料

1. GitHub 可下载代码：仓库 README 包含安装、数据下载、预训练权重、运行和复现命令。
2. 英文研究报告：`output/pdf/ELEC5305_Project_Report_Jiajun_Liu.pdf`，10页，含目录、图表、文献和 AI 辅助声明。
3. 学生录制的英文演示视频：随最终提交文件夹单独提供原始 MOV，不上传到 GitHub。保留原录屏内容。

GitHub 仓库：<https://github.com/JiajunLiu-June/elec5305-project-530878323>
仓库名中的历史数字不作学号使用；正确学号是550378323。

## 复现

使用 Python 3.12，从代码目录运行。完整命令和实验解释见 `README.md`。

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python tools/download_subset.py --tracks 5
python tools/cache_models.py
python -m src.demo
python -m src.study --help
```

`src.study` 是 `src.run_experiment` 的短命令入口，使用相同实现和参数。
`results/study/` 保存了已测量的CSV、图表和清单，不必先重跑实验才能查看结果。
原始数据、预训练权重和虚拟环境不应加入GitHub或作业提交包；按说明分别获取。

## 核心结论与范围

5首MUSDB18-HQ测试歌、9个有效片段、36个配对输入、252行主实验结果。
自然混音按歌曲等权汇总：IRM为11.69 dB，Clipped PSM为12.47 dB，约改善0.78 dB。
HTDemucs的一个短输入失败案例在加入上下文后由约0提升到8.84 dB。
PSM是既有方法；本项目贡献是复现、受控实验、消融与解释。模型训练数据不同，
样本较小，上下文和分块边界同时变化，不能据此声称架构本身普遍更优或听众偏好。
