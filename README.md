# MachineLarning_Numpy_CodeCraft2020

### 使用说明
#### 1. 导入项目
```bash
git clone https://github.com/MarkShawn2020/MachineLarning_Numpy_CodeCraft2020
pip install numpy==1.17.2
```


#### 2. 配置数据
打开`core/settings.py`文件，设定您的数据文件夹位置。

考虑到git的拉取速度，这些文件我们没有上传。

此外，您的本地程序生成的预测文件`result.txt`也会自动存储在该文件夹下。

#### 3. 测试程序
RUN `demo/Model_1_MBSGD.py`，预期输出：

![demo_model_1](doc/run_demo_model_1.png)

#### 4. 发布程序
如果您的程序测试通过，
可以使用`scripts/convert.py`文件将其自动转换成单文件版本，
而无需您手动修改包的导入，这是本项目最大的福利之一。

之后您可以直接将`release/Main.py`文件上传到服务器，
或者本地运行`python release/Main.py`。

Good Luck！


## 项目说明
本项目部分参考`PYTORCH`的框架设计。

目前已经实现：
- 基于生成器、自动切片、随机打乱、可自由分割训练集和验证集的DataLoader
- 继承于通用模型类的LogisticRegression类
- 一些常用的functions如交叉熵等
- 本地和服务器均可使用的路径配置
- 一些装饰器，如函数计时等

### 使用简单随机梯度下降Logistic测试结果
![eval_data](./doc/eval_data.png)

### TODO
- [ ] 融合一些高级优化器
- [ ] 加入其他模型
- [ ] 其他扩展与性能优化

### 最后声明
考虑到Python可能对最后的成绩没有什么决定性作用，
但大家可以一起学习研究，所以开源给大家使用
也欢迎大家积极贡献代码和issue，感谢~