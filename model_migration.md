# 模型分析

在进行正式的代码迁移前，需要对要进行迁移的代码做一些简单的分析，判断哪些代码可以直接复用，哪些代码必须迁移到MindSpore。

一般的，只有与硬件相关的代码部分必须要迁移到MindSpore，比如：

- 模型输入相关，包含模型参数加载，数据集包装等；
- 模型构建和执行的代码；
- 模型输出相关，包含模型参数保存等。

像Numpy，OpenCV等CPU上计算的三方库，Configuration，Tokenizer等不需要昇腾、GPU处理的python操作可以直接复用原始代码。

# 数据集包装

MindSpore提供了多种典型开源数据集的解析读取，如MNIST、CIFAR-10、CLUE、LJSpeech等，详情可参考[mindspore.dataset](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.html)。

## 自定义数据加载 GeneratorDataset

在迁移场景，最常用的数据加载方式是[GeneratorDataset](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset)，只需对python迭代器的简单包装，就可以直接对接MindSpore模型进行训练、推理。

```python
import numpy as np
from mindspore import dataset as ds

num_parallel_workers = 2  # 多线程/进程数
world_size = 1            # 并行场景使用，通信group_size
rank = 0                  # 并行场景使用，通信rank_id

class MyDataset:
    def __init__(self):
        self.data = np.random.sample((5, 2))
        self.label = np.random.sample((5, 1))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

dataset = ds.GeneratorDataset(source=MyDataset(), column_names=["data", "label"],
                              num_parallel_workers=num_parallel_workers, shuffle=True,
                              num_shards=1, shard_id=0)
train_dataset = dataset.batch(batch_size=2, drop_remainder=True, num_parallel_workers=num_parallel_workers)
```

一个典型的数据集构造如上：构造一个python类，必须有__getitem__和__len__方法，分别表示每一步迭代取的数据和整个数据集遍历一次的大小，其中index表示每次取数据的索引，当shuffle=False时按顺序递增，当shuffle=True时随机打乱。

GeneratorDataset至少需要包含：

- source：一个python迭代器；
- column_names：迭代器__getitem__方法每个输出的名字。

此外，还有一些常用的配置：

- num_parallel_workers：GeneratorDataset多进程并行处理的进程数；
- shuffle：是否随机打乱；
- num_shards：并行场景配合shard_id使用，数据切片个数
- shard_id：并行场景配合num_shards使用，数据切片id

更多使用方法参考[GeneratorDataset](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset)。

dataset.batch将数据集中连续batch_size条数据组合为一个批数据，至少需要包含：

- batch_size：指定每个批处理数据包含的数据条目

此外，还有一些常用的配置：

- drop_remainder：当最后一个批处理数据包含的数据条目小于 batch_size 时，是否将该批处理丢弃；
- num_parallel_workers：batch操作多线程并行处理的线程数；

更多使用方法参考[Dataset.batch](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/dataset_method/batch/mindspore.dataset.Dataset.batch.html)。

## 与PyTorch数据集构建差别

![generatordataset_dataloader.png](generatordataset_dataloader.png)

MindSpore的GeneratorDataset与PyTorch的DataLoader的主要差别有：

- MindSpore的GeneratorDataset必须传入column_names；
- PyTorch的数据增强输入的对象是Tensor类型，MindSpore的数据增强输入的对象是numpy类型，且数据处理不能用MindSpore的计算算子；
- PyTorch的batch操作是DataLoader的属性，MindSpore的batch操作是独立的方法。

详细可参考[与torch.utils.data.DataLoader的差异](https://www.mindspore.cn/docs/zh-CN/r2.4.0/note/api_mapping/pytorch_diff/DataLoader.html)。

# 模型构建

## 网络基本构成单元 Cell

MindSpore的网络搭建主要使用Cell进行图的构造，用户需要定义一个类继承Cell这个基类，在init里声明需要使用的API及子模块，在construct里进行计算：

<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch </td> <td style="text-align:center"> MindSpore </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
import torch

class MyCell_pt(torch.nn.Module):
    def __init__(self, forward_net):
        super(MyCell_pt, self).__init__()
        self.net = forward_net

    def forward(self, x):
        y = self.net(x)
        return torch.nn.functional.relu(y)

inner_net_pt = torch.nn.Conv2d(120, 240, kernel_size=4, bias=False)
pt_net = MyCell_pt(inner_net_pt)
for i in pt_net.parameters():
    print(i)
```
</pre>
</td>
<td style="vertical-align:top"><pre>

```python
from mindspore import mint, nn

class MyCell(nn.Cell):
    def __init__(self, forward_net):
        super(MyCell, self).__init__(auto_prefix=True)
        self.net = forward_net

    def construct(self, x):
        y = self.net(x)
        return mint.nn.functional.relu(y)

inner_net = mint.nn.Conv2d(120, 240, kernel_size=4, bias=False)
my_net = MyCell(inner_net)
for i in my_net.trainable_params():
    print(i)
```
</pre>
</td>
</tr>
</table>

MindSpore和PyTorch构建模型的方法差不多，使用算子的差别可以参考[API差异文档](https://www.mindspore.cn/docs/zh-CN/r2.4.0/note/api_mapping/pytorch_diff/Conv2d.html)。

### 模型保存和加载

PyTorch提供了 `state_dict()` 用于参数状态的查看及保存，`load_state_dict` 用于模型参数的加载。

MindSpore的优化器模块继承自 `Cell`，使用 `save_checkpoint` 与`load_checkpoint` 。

<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch </td> <td style="text-align:center"> MindSpore </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
# 使用torch.save()把获取到的state_dict保存到pkl文件中
torch.save(pt_model.state_dict(), save_path)

# 使用torch.load()加载保存的state_dict，
# 然后使用load_state_dict将获取到的state_dict加载到模型中
state_dict = torch.load(save_path)
pt_model.load_state_dict(state_dict)
```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
# 模型权重保存：
ms.save_checkpoint(ms_model, save_path)

# 使用ms.load_checkpoint()加载保存的ckpt文件，
# 然后使用load_param_into_net将获取到的param_dict加载到模型中
param_dict = ms.load_checkpoint(save_path)
ms.load_param_into_net(ms_model, param_dict)
```
</pre>
</td>
</tr>
</table>

### 单元测试

为了保证构建的MindSpore的Cell迁移正确，需要使用相同的输入数据和参数，对输出做比较：

```python
import numpy as np
import mindspore as ms
from mindspore import ops, nn
import torch

def get_pt2ms_mappings(m):
    mappings = {}  # pt_param_name: (ms_param_name, pt_param_to_ms_param_func)
    for name, cell in m.cells_and_names():
        if isinstance(cell, (nn.Conv1d, nn.Conv1dTranspose)):
            mappings[f"{name}.weight"] = f"{name}.weight", lambda x: ms.Parameter(
                ops.expand_dims(x, axis=-2), name=f"{name}.weight"
            )
        elif isinstance(cell, nn.Embedding):
            mappings[f"{name}.weight"] = f"{name}.embedding_table", lambda x: x
        elif isinstance(cell, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            mappings[f"{name}.weight"] = f"{name}.gamma", lambda x: x
            mappings[f"{name}.bias"] = f"{name}.beta", lambda x: x
            if isinstance(cell, (nn.BatchNorm2d,)):
                mappings[f"{name}.running_mean"] = f"{name}.moving_mean", lambda x: x
                mappings[f"{name}.running_var"] = f"{name}.moving_variance", lambda x: x
                mappings[f"{name}.num_batches_tracked"] = None, lambda x: x
    return mappings

def convert_state_dict(m, state_dict_pt):
    dtype_mappings = {
        torch.float16: ms.float16,
        torch.float32: ms.float32,
        torch.bfloat16: ms.bfloat16,
    }

    mappings = get_pt2ms_mappings(m)
    state_dict_ms = {}
    for name_pt, data_pt in state_dict_pt.items():
        name_ms, data_mapping = mappings.get(name_pt, (name_pt, lambda x: x))
        data_ms = ms.Parameter(
            data_mapping(ms.Tensor.from_numpy(data_pt.float().numpy()).to(dtype_mappings[data_pt.dtype])), name=name_ms
        )
        if name_ms is not None:
            state_dict_ms[name_ms] = data_ms
    return state_dict_ms

my_net = MSNet()
pt_net = PTNet()

ms.load_param_into_net(my_net, convert_state_dict(my_net, pt_net.state_dict()), strict_load=True)

# 构造随机输入
x = np.random.uniform(-1, 1, (2, 120, 12, 12)).astype(np.float32)

y_ms = my_net(ms.Tensor(x))
y_pt = pt_net(torch.from_numpy(x))
diff = np.max(np.abs(y_ms.asnumpy() - y_pt.detach().numpy()))
print(diff)
```

**在迁移Cell的过程中最好对每个Cell都做一次单元测试，保证迁移的一致性。**

## 损失函数

在阅读本章节之前，请先阅读MindSpore官网教程[损失函数](https://www.mindspore.cn/docs/zh-CN/master/model_train/custom_program/loss.html)。

MindSpore官网教程损失函数中讲解了内置、自定义和多标签损失函数，以及在模型训练中的使用指导。这里就MindSpore的损失函数与PyTorch的损失函数在功能和接口差异方面给出差异列表。

| torch.nn | torch.nn.functional | mindspore.nn | mindspore.ops | 差异说明 |
| -------- | ------------------- | ------------ | ------------- | ------- |
| torch.nn.L1Loss | torch.nn.functional.l1_loss | mindspore.nn.L1Loss| mindspore.ops.l1_loss| 一致 |
| torch.nn.MSELoss | torch.nn.functional.mse_loss | mindspore.nn.MSELoss| mindspore.ops.mse_loss| 一致 |
| torch.nn.CrossEntropyLoss | torch.nn.functional.cross_entropy | mindspore.nn.CrossEntropyLoss| mindspore.ops.cross_entropy| [nn接口差异](https://www.mindspore.cn/docs/zh-CN/r2.4.0/note/api_mapping/pytorch_diff/CrossEntropyLoss.html) |
| torch.nn.CTCLoss | torch.nn.functional.ctc_loss | mindspore.nn.CTCLoss| mindspore.ops.ctc_loss| 一致 |
| torch.nn.NLLLoss | torch.nn.functional.nll_loss | mindspore.nn.NLLLoss| mindspore.ops.nll_loss| 一致 |
| torch.nn.PoissonNLLLoss | torch.nn.functional.poisson_nll_loss | mindspore.nn.PoissonNLLLoss| - | 一致 |
| torch.nn.GaussianNLLLoss | torch.nn.functional.gaussian_nll_loss | mindspore.nn.GaussianNLLLoss| mindspore.ops.gaussian_nll_loss | 一致 |
| torch.nn.KLDivLoss | torch.nn.functional.kl_div | mindspore.nn.KLDivLoss| mindspore.ops.kl_div| MindSpore不支持 `log_target` 参数 |
| torch.nn.BCELoss | torch.nn.functional.binary_cross_entropy | mindspore.nn.BCELoss| mindspore.ops.binary_cross_entropy| 一致 |
| torch.nn.BCEWithLogitsLoss | torch.nn.functional.binary_cross_entropy_with_logits | mindspore.nn.BCEWithLogitsLoss| mindspore.ops.binary_cross_entropy_with_logits| 一致 |
| torch.nn.MarginRankingLoss | torch.nn.functional.margin_ranking_loss | mindspore.nn.MarginRankingLoss| mindspore.ops.margin_ranking_loss | 一致 |
| torch.nn.HingeEmbeddingLoss | torch.nn.functional.hinge_embedding_loss | mindspore.nn.HingeEmbeddingLoss| mindspore.ops.hinge_embedding_loss | 一致 |
| torch.nn.MultiLabelMarginLoss | torch.nn.functional.multilabel_margin_loss | mindspore.nn.MultiLabelMarginLoss | mindspore.ops.multilabel_margin_loss| 一致 |
| torch.nn.HuberLoss | torch.nn.functional.huber_loss | mindspore.nn.HuberLoss | mindspore.ops.huber_loss| 一致 |
| torch.nn.SmoothL1Loss | torch.nn.functional.smooth_l1_loss | mindspore.nn.SmoothL1Loss | mindspore.ops.smooth_l1_loss| 一致 |
| torch.nn.SoftMarginLoss | torch.nn.functional.soft_margin_loss | mindspore.nn.SoftMarginLoss| mindspore.ops.soft_margin_loss | 一致 |
| torch.nn.MultiLabelSoftMarginLoss | torch.nn.functional.multilabel_soft_margin_loss | mindspore.nn.MultiLabelSoftMarginLoss| mindspore.ops.multilabel_soft_margin_loss| 一致 |
| torch.nn.CosineEmbeddingLoss | torch.nn.functional.cosine_embedding_loss | mindspore.nn.CosineEmbeddingLoss| mindspore.ops.cosine_embedding_loss| 一致 |
| torch.nn.MultiMarginLoss | torch.nn.functional.multi_margin_loss | mindspore.nn.MultiMarginLoss | mindspore.ops.multi_margin_loss | 一致 |
| torch.nn.TripletMarginLoss | torch.nn.functional.triplet_margin_loss | mindspore.nn.TripletMarginLoss| mindspore.ops.triplet_margin_loss | [功能一致，参数个数或顺序不一致](https://www.mindspore.cn/docs/zh-CN/r2.4.0/note/api_mapping/pytorch_diff/TripletMarginLoss.html) |
| torch.nn.TripletMarginWithDistanceLoss | torch.nn.functional.triplet_margin_with_distance_loss | mindspore.nn.TripletMarginWithDistanceLoss | - | 一致 |

## 优化器

PyTorch和MindSpore同时支持的优化器异同比较详见[API映射表](https://mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_api_mapping.html#torch-optim)。MindSpore暂不支持的优化器：LBFGS、NAdam、RAdam。

### 优化器的执行和使用差异

PyTorch单步执行优化器时，一般需要手动执行 `zero_grad()` 方法将历史梯度设置为 ``0`` （或 ``None`` ），然后使用 `loss.backward()` 计算当前训练step的梯度，最后调用优化器的 `step()` 方法实现网络权重的更新；

使用MindSpore中的优化器时，只需要直接对梯度进行计算，然后使用 `optimizer(grads)` 执行网络权重的更新。

<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch </td> <td style="text-align:center"> MindSpore </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = ExponentialLR(optimizer, gamma=0.9)

optimizer.zero_grad()
output = model(input)
loss = loss_fn(output, target)
loss.backward()
optimizer.step()
scheduler.step()
```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
import mindspore
from mindspore import nn

lr = nn.exponential_decay_lr(0.01, decay_rate, total_step, step_per_epoch, decay_epoch)

optimizer = nn.SGD(model.trainable_params(), learning_rate=lr, momentum=0.9)
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
(loss, _), grads = grad_fn(data, label)
# 在优化器里自动做学习率更新
optimizer(grads)
```

</pre>
</td>
</tr>
</table>

### 学习率策略对比

PyTorch中定义了 `LRScheduler` 类用于对学习率进行管理。使用动态学习率时，将 `optimizer` 实例传入 `LRScheduler` 子类中，通过循环调用 `scheduler.step()` 执行学习率修改，并将修改同步至优化器中。

MindSpore中的动态学习率有 `Cell` 和 `list` 两种实现方式，两种类型的动态学习率使用方式一致，都是在实例化完成之后传入优化器，前者在内部的 `construct` 中进行每一步学习率的计算，后者直接按照计算逻辑预生成学习率列表，训练过程中内部实现学习率的更新。具体请参考[动态学习率](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.nn.html#%E5%8A%A8%E6%80%81%E5%AD%A6%E4%B9%A0%E7%8E%87)。

## 自动微分

MindSpore 和 PyTorch 都提供了自动微分功能，让我们在定义了正向网络后，可以通过简单的接口调用实现自动反向传播以及梯度更新。但需要注意的是，MindSpore 和 PyTorch 构建反向图的逻辑是不同的，这个差异也会带来 API 设计上的不同。

<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch的自动微分 </td> <td style="text-align:center"> MindSpore的自动微分 </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
# torch.autograd:
# backward是累计的，更新完之后需清空optimizer

import torch.nn as nn
import torch.optim as optim

# 实例化模型和优化器
model = PT_Model()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 定义损失函数：均方误差（MSE）
loss_fn = nn.MSELoss()

# 前向传播：计算模型输出
y_pred = model(x)

# 计算损失：将预测值与真实标签计算损失
loss = loss_fn(y_pred, y_true)

# 反向传播：计算梯度
loss.backward()
# 优化器更新
optimizer.step()
```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
# ms.grad:
# 使用grad接口，输入正向图，输出反向图
import mindspore as ms
from mindspore import nn

# 实例化模型和优化器
model = MS_Model()
optimizer = nn.SGD(model.trainable_params(), learning_rate=0.01)

# 定义损失函数：均方误差（MSE）
loss_fn = nn.MSELoss()

def forward_fn(x, y_true):
    # 前向传播：计算模型输出
    y_pred = model(x)
    # 计算损失：将预测值与真实标签计算损失
    loss = loss_fn(y_pred, y_true)
    return loss, y_pred

# 计算loss和梯度
grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
(loss, _), grads = grad_fn(x, y_true)
# 优化器更新
optimizer(grads)
```

</pre>
</td>
</tr>
</table>

## 模型训练和推理

下面是一个在MindSpore的Trainer的例子，包含了训练和训练时推理：

```python
import mindspore as ms
from mindspore import nn
from mindspore.amp import StaticLossScaler, all_finite

class Trainer:
    """一个有两个loss的训练示例"""
    def __init__(self, net, loss1, loss2, optimizer, train_dataset, loss_scale=1.0, eval_dataset=None, metric=None):
        self.net = net
        self.loss1 = loss1
        self.loss2 = loss2
        self.opt = optimizer
        self.train_dataset = train_dataset
        self.train_data_size = self.train_dataset.get_dataset_size()    # 获取训练集batch数
        self.weights = self.opt.parameters
        # 注意value_and_grad的第一个参数需要是需要做梯度求导的图，一般包含网络和loss。这里可以是一个函数，也可以是Cell
        self.value_and_grad = ms.value_and_grad(self.forward_fn, None, weights=self.weights, has_aux=True)

        # 分布式场景使用
        self.grad_reducer = self.get_grad_reducer()
        self.loss_scale = StaticLossScaler(loss_scale)
        self.run_eval = eval_dataset is not None
        if self.run_eval:
            self.eval_dataset = eval_dataset
            self.metric = metric
            self.best_acc = 0

    def get_grad_reducer(self):
        grad_reducer = nn.Identity()
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        # 判断是否是分布式场景，分布式场景的设置参考上面通用运行环境设置
        reducer_flag = (parallel_mode != ms.ParallelMode.STAND_ALONE)
        if reducer_flag:
            grad_reducer = nn.DistributedGradReducer(self.weights)
        return grad_reducer

    def forward_fn(self, inputs, labels):
        """正向网络构建，注意第一个输出必须是最后需要求梯度的那个输出"""
        logits = self.net(inputs)
        loss1 = self.loss1(logits, labels)
        loss2 = self.loss2(logits, labels)
        loss = loss1 + loss2
        loss = self.loss_scale.scale(loss)
        return loss, loss1, loss2

    @ms.jit    # jit加速，需要满足图模式构建的要求，否则会报错
    def train_single(self, inputs, labels):
        (loss, loss1, loss2), grads = self.value_and_grad(inputs, labels)
        loss = self.loss_scale.unscale(loss)
        grads = self.loss_scale.unscale(grads)
        grads = self.grad_reducer(grads)
        state = all_finite(grads)
        if state:
            self.opt(grads)

        return loss, loss1, loss2

    def train(self, epochs):
        train_dataset = self.train_dataset.create_dict_iterator(num_epochs=epochs)
        self.net.set_train(True)
        for epoch in range(epochs):
            # 训练一个epoch
            for batch, data in enumerate(train_dataset):
                loss, loss1, loss2 = self.train_single(data["image"], data["label"])
                if batch % 100 == 0:
                    print(f"step: [{batch} /{self.train_data_size}] "
                          f"loss: {loss}, loss1: {loss1}, loss2: {loss2}", flush=True)
            # 保存当前epoch的模型和优化器权重
            ms.save_checkpoint(self.net, f"epoch_{epoch}.ckpt")
            ms.save_checkpoint(self.opt, f"opt_{epoch}.ckpt")
            # 推理并保存最好的那个checkpoint
            if self.run_eval:
                eval_dataset = self.eval_dataset.create_dict_iterator(num_epochs=1)
                self.net.set_train(False)
                self.eval(eval_dataset, epoch)
                self.net.set_train(True)

    def eval(self, eval_dataset, epoch):
        self.metric.clear()
        for batch, data in enumerate(eval_dataset):
            output = self.net(data["image"])
            self.metric.update(output, data["label"])
        accuracy = self.metric.eval()
        print(f"epoch {epoch}, accuracy: {accuracy}", flush=True)
        if accuracy >= self.best_acc:
            # 保存最好的那个checkpoint
            self.best_acc = accuracy
            ms.save_checkpoint(self.net, "best.ckpt")
            print(f"Updata best acc: {accuracy}")
```

### 分布式训练

以数据并行为例，首先指定运行模式、硬件设备等，通过init()初始化HCCL、NCCL或MCCL通信域。

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
init()
ms.set_seed(1)

# 模型定义与Trainer构建同上
...
trainer = Trainer(...)
trainer.train()
```

准备启动脚本:

```shell
# 单机8卡
msrun --worker_num=8 --local_worker_num=8 net.py
```

更多细节详见[msrun](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/parallel/msrun_launcher.html)。

### 离线推理

除了可以在线推理外，MindSpore提供了很多离线推理的方法适用于不同的环境，详情请参考[模型推理](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_infer/ms_infer/llm_inference_overview.html)。
