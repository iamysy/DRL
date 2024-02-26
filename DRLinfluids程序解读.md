# DRLinfluids程序解读

个人解读，如有不足请指出，共同改进。

[TOC]



## 主要含有的内容

文件夹DRLinFluids涉及对函数运行相关设置。

外部文件是对一些简略参数进行修改。

### __init__文件

__init__.py文件包含对整个程序进行封装,其与DRLinFluids作用相同，DRLinFluids同样需要设置。

```python
from gym.envs.registration import register
from DRLinFluids import environments_tianshou
from DRLinFluids import utils,cfd
from DRLinFluids.environments_tianshou import OpenFoam_tianshou
```

他对该框架内的所有模块进行调用

```python
__all__ = [
    "OpenFoam_tianshou",
]
```

该用法表示使用`import 函数名`或者`from 模块名 import 函数名`都可以调用不同函数，当使用`from 模块名 import *`时不可以使用all定义外的函数

```python
register(
    id="OpenFoam-v0",
    entry_point="airfoil2D.DRLinFluids:OpenFoam_tianshou", # 第一个myenv是文件夹名字，第二个myenv是文件名字，MyEnv是文件内类的名字
    max_episode_steps=100,
    #reward_threshold=100.0,
)
```

封装一个id为OpenFoam-v0的文件，此处`max_episode_steps`对其进行更改即可

### envobject_airfoil.py文件

envobject_airfoil.py对奖励函数形式与智能体动作进行设置

#### 类FlowAroundAirfoil2D

```python
import numpy as np
from scipy import signal
from DRLinFluids.environments_tianshou import OpenFoam_tianshou
```

```python
class FlowAroundAirfoil2D(OpenFoam_tianshou):
    def reward_function(self):
        vortex_shedding_period = 0.05
        drug_coeffs_sliding_average = self.force_coeffs_sliding_average(vortex_shedding_period)[0]
        lift_coeffs_sliding_average = self.force_coeffs_sliding_average(vortex_shedding_period)[1]
        print(0 - drug_coeffs_sliding_average - 0.286,0.2 * np.abs(lift_coeffs_sliding_average - 0.88516))
        return   0 - drug_coeffs_sliding_average - 0.286 - 0.2 * np.abs(lift_coeffs_sliding_average - 0.88516)


    def agent_actions_decorator(self, actions):
        if self.num_trajectory < 1.5:
            new_action = 0.4 * actions
        else:
            new_action = np.array(self.decorated_actions_sequence[self.num_trajectory - 2]) \
                         + 0.4 * (np.array(actions) - np.array(self.decorated_actions_sequence[self.num_trajectory - 2]))
            # new_action = 0.1 * actions
        return new_action
        # return np.array((1, ))
```

类`FlowAroundAirfoil2D()`调用了`environment_tianshou.py`内的`OpenFoam_tianshou`类

##### 函数reward_function

其中`reward_function`函数继承`OpenFoam_tianshou`内的参数

`vortex_shedding_period`在此设定为0.05，其等于foam_params中的`interaction_period`，都代表模型的涡脱落周期，其后续会在`environment_tianshou.py`进一步进行设定。

```python
drug_coeffs_sliding_average = self.force_coeffs_sliding_average(vortex_shedding_period)[0]
```

此处引用了`environment_tianshou.py`内类OpenFoam_tianshou中函数`force_coeffs_sliding_average()`,其在文件中的定义如下

```python
def force_coeffs_sliding_average(self, sliding_time_interval):
   # sampling_num用来计算采样点的数量
   sampling_num = int(sliding_time_interval / self.foam_params['delta_t'])
   # 计算涡流脱落周期内升力系数的滑动平均值
   if self.history_force_Coeffs_df.shape[0] <= sampling_num:
   # if用来判断history矩阵的行数（一个矩阵行数代表一个时间步的阻力升力系数）是否小于等于采样点数量
   # 对提取出的数据进行平均获得一个较为平缓的数据
   # 用来计算奖励函数的cl与cd是在一定时间步内的平均值
      sliding_average_cd = np.mean(signal.savgol_filter(self.history_force_Coeffs_df.iloc[:, 2], 25, 0))
      sliding_average_cl = np.mean(signal.savgol_filter(self.history_force_Coeffs_df.iloc[:, 3], 25, 0))
   else:
      sliding_average_cd = np.mean(
         signal.savgol_filter(self.history_force_Coeffs_df.iloc[-sampling_num:, 2], 25, 0))
      sliding_average_cl = np.mean(
         signal.savgol_filter(self.history_force_Coeffs_df.iloc[-sampling_num:, 3], 25, 0))
   return sliding_average_cd, sliding_average_cl
```

###### 涡脱落频率的计算方法总结#存在问题

大致如下：

1.在openfoam算例中，算得升力系数，其中一种方法是在`controlDict`内添加下列内容来计算算例的升力系数。

```c++
functions
{
    forceCoeffs
    {
        type forceCoeffs;
        functionObjectLibs ("libforces.so");
        patches (cylinder);
        rho rhoInf;
        rhoInf 1.225;
        CofR (0 0 0);
        liftDir (0 1 0);
        dragDir (1 0 0);
        pitchAxis (0 0 1);
        magUInf 10;
        lRef 1;
        Aref 1;
    }
}
```

或者是通过本case中的方法，在`system`中添加`forceCoeffsIncompressible`文件，文件内有对升阻力系数计算的方法，计算升阻力系数的区域是机翼表面。

2.将postprocessing内的forceCoeffsIncompressible中的计算结果导入到matlab中，导入的方法根据不同的文件类型可以使用不同的语句,读取的内容必须是数值矩阵

```matlab
# dat文件
X = importdata('data.dat');
# csv文件
X = readtable('liftCoeffs.csv');
# xlsx文件
X = readmatrix('data.xlsx');
```

```matlab
% 读取整个文件为单元数组
C = readcell('C:\Users\yousunyu\Desktop\try_test\forceCoeffs.dat');
% 提取10行以后的所有行以及第1列和第4列的数据到一个变量
X = C(10:end,[1 4]);
% 将单元数组转换为数值矩阵
X = cell2mat(X);
% 选择第一列作为信号
signal = X(:,1);
% 对信号进行快速傅里叶变换,在matlab中，使用fft函数对矩阵对象进行快速傅里叶变换，得到一个复数向量，此处只需要升力系数列数据做fft，并不需要带有时间列，此为频域信号向量：
Y = fft(signal);
% 计算信号长度
n = length(signal);
% 计算采样率（假设信号持续时间为10秒），此处为我所应用的数据的采样间隔（升力的计算间隔即controlDict内的writeinterval设定的时间）
Fs = n/10;
% 生成频率向量
f = linspace(0,Fs,n);
% 计算功率谱,
P = abs(Y).^2/n;
% 找到功率谱中的最大值和对应的索引
[M,I] = max(P);
```

在matlab中，使用plot函数绘制快速傅里叶变换的结果，显示频率与振幅的关系：

```matlab
plot(f,abs(Y)); % 绘制频谱图
xlabel('Frequency (Hz)'); % 设置横轴标签
ylabel('Amplitude'); % 设置纵轴标签
```

同时根据索引计算脱落频率，涡脱落频率等于索引乘采样频率除以升力系数向量的长度
$$
涡脱落频率=(I*采样频率)/n
$$
其中I为最大值对应的索引，n为升力系数向量数据的长度。

###### 平滑函数应用方法

该函数中含有scipy.signal.savgol_filter(x, window_length, polyorder)

其中

- x为要滤波的信号

- window_length即窗口长度
  取值为奇数且不能超过len(x)。它越大，则平滑效果越明显；越小，则更贴近原始曲线。

- polyorder为多项式拟合的阶数。
  它越小，则平滑效果越明显；越大，则更贴近原始曲线。

  在本例中self.history_force_Coeffs_df.iloc[:, 2]为索要滤波的信号，窗口长度为25，此处的窗口长度一定要小于我们所设置的samping_num（涡脱落是几倍的时间步），polyorder设置为0表示需要很大的平滑效果。

该平滑曲线语句需要**x为数据类型**，不论是array还是dataframe都可以，且读取的**数据len应该一致**，window_lenth必须是正奇数。

参考：[(34条消息) Python 生成曲线进行快速平滑处理_window_length must be odd._智能音箱设计的博客-CSDN博客](https://blog.csdn.net/weixin_43821212/article/details/100016021)

平滑函数并非插值法，而是应用Savitzky-Golay 滤波器，该滤波器是对一定长度窗口内的数据点进行K阶多项式拟合。

参考：[(36条消息) Savitzky-Golay平滑滤波的python实现_sg滤波的窗口如何选择_假小牙的博客-CSDN博客](https://blog.csdn.net/sinat_21258931/article/details/79298478)

##### agent_actions_decorator()函数

对动作进行smooth公式为：
$$
A1=0.4A1'						(n=1)
$$

$$
A1=[A0+0.4(A1'-A0)](n>1)
$$

此处有上标的为Agent给出的动作，A1为输入至环境的动作，即对射流速度的控制。

### launch_multiprocessing_training_airfoil.py文件

1. 首先是对库的引用
2. 对参数、超参数进行定义设置
3. 设置foam参数以及智能体参数
4. 自定义环境
5. 设置DRL所使用的算法
6. 启动训练

```python
import re
import datetime
import envobject_airfoil
import argparse
import os
import numpy as np
import torch
import pprint
from torch.utils.tensorboard import SummaryWriter
import gym

from tianshou.data import VectorReplayBuffer,AsyncCollector
from tianshou.policy import SACPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
```

上半部分是一些python基本文件的引用

下半部分是tianshou强化学习库的引用，利用tianshou库就不需要进行算法的编辑，直接应用即可。

##### get_args()函数

通过args来对一些参数、超参数进行定义以及设置

###### 网络、算法超参数：

|             超参数             |   value    |
| :----------------------------: | :--------: |
|       折扣率（γ）(gamma)       |    0.99    |
|  网络更新时的平滑参数（tau）   |   0.005    |
|      熵权重系数（alpha）       |    0.2     |
|           学习率(lr)           |   0.001    |
|   缓冲区大小（buffer-size）    |   20000    |
| 神经网络隐藏层（hidden_sizes） | [512, 512] |



```python
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='OpenFoam-v0')
    parser.add_argument('--reward-threshold', type=float, default=15.8)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--il-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--auto-alpha', type=int, default=1)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument('--step-per-epoch', type=int, default=100)
    parser.add_argument('--il-step-per-epoch', type=int, default=1)
    parser.add_argument('--step-per-collect', type=int, default=20)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[512,512])
    parser.add_argument(
        '--imitation-hidden-sizes', type=int, nargs='*', default=[512, 512]
    )
    # 此处对DRL并行环境个数进行设定，每个并行环境内容相同，同时进行训练，并将训练后的内容汇聚
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--rew-norm', action="store_true", default=False)
    parser.add_argument('--n-step', type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--save-buffer-name", type=str, default=None)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="atari.benchmark")
    parser.add_argument(
        "--icm-lr-scale",
        type=float,
        default=0.,
        help="use intrinsic curiosity module with this lr scale"
    )
    # 是否需要利用已经保存好的训练模型继续开始训练
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="restart"
    )
    shell_args = vars(parser.parse_args())
    args = parser.parse_known_args()[0]

    shell_args['num_episodes']=500
    shell_args['max_episode_timesteps']=150
    return args
```

改变其中参数、超参数可以对整个框架进行修改。

##### test_sac_with_il()函数

他调用上述所加入的参数

```python
foam_params = {
    # 此处定义openfoam中时间步
    'delta_t': 0.001,
    # 此处定义openfoam中所用的求解器
    'solver': 'pimpleFoam',
    # 此处定义openfoam中并行数量
    'num_processor': 5,
    'of_env_init': 'source ~/OpenFOAM/OpenFOAM-8/etc/bashrc',
    # 初始化流场，初始化state,更加明显的对比
    'cfd_init_time': 0.1,  
    'num_dimension': 2,
    'verbose': False
}
# 对不同动作进行不同定义
"""
# 如果想增加控制数量可以从此处添加，相应的需要在agent_params中添加对你所添加动作的定义
# 此处即为对修改位置的定义，此后会在environment_tianshou.py内配合utils文件对openfoam中U内容进行更换，具体内容可配合environment_tianshou.py中step函数进行理解
entry_dict_q0 = {
    'U': {
        'JET': {
            'q0': '{x1}',
        }
    }
}
# 在此处添加完之后，在agent_params内添加
# 'variables_q0': ('x1',),
"""
entry_dict_q0 = {
    'U': {
        'JET': {
            'q0': '{x}',
        }
    }
}

entry_dict_q1 = {
    'U': {
        'JET': {
            'q1': '{y}',
        }
    }
}

entry_dict_t0 = {
    'U': {
        'JET': {
            't0': '{t}'
        }
    }
}

agent_params = {
    'entry_dict_q0': entry_dict_q0,
    'entry_dict_q1': entry_dict_q1,
    'entry_dict_t0': entry_dict_t0,
    'deltaA': 0.05,
    'minmax_value': (-4, 4),
    # 定义交互时间，此处值可以参考涡脱落时间设置
    'interaction_period': 0.025,
    'purgeWrite_numbers': 0,
    # 对于openfoam中写入openfoam文件时间设置
    'writeInterval': 0.025,
    # 时间步
    'deltaT': 0.001,
    'variables_q0': ('x',),
    'variables_q1': ('y',),
    'variables_t0': ('t',),
    'verbose': False,
    "zero_net_Qs": True,
}
# 设置状态空间所取的值
state_params = {
    'type': 'velocity'
}
root_path = os.getcwd()
env_name_list = sorted([envs for envs in os.listdir(root_path) if re.search(r'^env\d+$', envs)])
env_path_list = ['/'.join([root_path, i]) for i in env_name_list]
# 创建测试模型环境
env = envobject_airfoil.FlowAroundAirfoil2D(
    foam_root_path=env_path_list[0],
    foam_params=foam_params,
    agent_params=agent_params,
    state_params=state_params,
)
```

`foam_params`设置时间步、求解器、并行核数、初始时间。

`entry_dict_*`定义动作形式，将动作输出接口与射流口相链接，此处由于openfoam中选用不同的输出方式，所选用的形式不同，例如：

前者`'q1': '{y}',`以这种方式定义的射流形式对应于下列的速度边界条件

```c++
JET
{
    type            incrementalJetParabolicVelocity;
    # 表示旋转角速度，单位为弧度每秒
    omega           0.17453293;
    # 管道半径，单位为米
    r               0.05;
    # 初始角度，单位为弧度
    theta0          1.5707963;
    # 初始流量，立方米每秒
    q0              0;
    # 最终流量，立方米每秒
    q1              0;
    # 管道壁面摩擦系数
    alpha           0.1;
    # 初始时间，单位为秒
    t0              0;
    # 时间步长
    deltaT          0.0005;
    # 初始速度场
    value           uniform (0 0 0);
}
```

若将`entry_dict_*`定义内的形式改为`'v0': '(0 {x} 0)'`则边界条件u中应该改为

```c++
    JET
    {
        type            incrementalJetUniVelocity;
        # 速度场初始值
        v0              (0 0 0);
        # 速度场最终值
        v1              (0 0 0);
        # 速度场松弛因子
        alpha           0.2;
        # 初始时间
        t0              0;
        # 时间步长
        deltaT          0.0005;
        # 初始速度场
        value           uniform (0 0 0);
    }
```

当引用不同类型的射流时，要注意在controlDict内加入libs

```
libs ("libinletParabolicVelocity.so" "libjetParabolicVelocity.so" "libincrementalJetParabolicVelocity.so" "libincrementalJetUniVelocity.so");
```

在`agent_params`内进行一些智能体参数设置

设置射流的上下限使训练更快收敛，设置`interaction_period`同上述，可以参考涡脱落周期，以及一些openfoam设置。

探针获取的值在压力值与速度值中选择速度值。

`root_path`用来获取当前所在位置，在本案例docker中指的是:/DRLinfluids/*(case folder name)

`env_name_list`首先由`envs`遍历路径下所有文件名称，每次遍历过程中配合`if`语句并且搭配`re.search`语句（匹配整个列表中由env+数字构成的字符串，并返回第一个成功的匹配。），并返回第一个成功的匹配，再有sorted语句对所获取的字符串进行排序，默认为升序。（ASCII的大小）

`env_path_list`应用join构建一个字符串表示每个并行环境的路径

`env`调用模块`envobject_square`内类`FlowAroundAirfoil2D`并且将参数输入。

```
# 在for循环中，利用索引值获取相应的值，遵循左闭右开即0-(training_num-1)
train_envs = SubprocVectorEnv(
    [lambda x=i: gym.make(args.task,foam_root_path=x,
                          foam_params=foam_params,
                          agent_params=agent_params,
                          state_params=state_params,
                          ) for i in env_path_list[0:args.training_num]],
    wait_num=args.training_num, timeout=0.2
)
# # test_envs = gym.make(args.task)
test_envs = SubprocVectorEnv(
    [lambda x=i: gym.make(args.task,foam_root_path=x,
                          foam_params=foam_params,
                          agent_params=agent_params,
                          state_params=state_params,
                          size=x, sleep=x) for i in env_path_list[args.training_num:(args.training_num+args.test_num)]]
)
```

`train_envs`应用`subprovectorenv`tianshou库内定义的并行环境函数，应用`lambda`函数对每个并行环境进行设置，并且将参数放入环境中对环境进行编译，`test_envs`同理。

```
args.state_shape = env.state_space.shape or env.state_space.n
args.action_shape = env.action_space.shape or env.action_space.n
args.max_action = env.action_space.high[0]
if args.reward_threshold is None:
    default_reward_threshold = {"OpenFoam-v0": 10, "Pendulum-v1": -250}
    args.reward_threshold = default_reward_threshold.get(
        args.task, env.spec.reward_threshold
    )
```

对状态空间以及动作空间进行定义。

对奖励门槛进行设置，如果未设置则按照默认的奖励门槛设置，`.get()`函数用来获取缓冲区内的一批数据。

```
# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
train_envs.seed([1, 2, 3, 4, 5])
```

对种子数的设置。

对强化学习中种子数的定义：在强化学习中，种子是指用于初始化随机数生成器的数。随机数生成器是一个函数，它生成用于各种目的的随机数序列，例如采样操作、探索环境、混洗数据等。通过将种子设置为固定值，可以确保随机数生成器每次被调用时都生成相同的数字序列。这有助于强化学习实验的重复性和调试，此处将种子数设置为1-5的数。

```
# model
net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
actor = ActorProb(
    net,
    args.action_shape,
    max_action=args.max_action,
    device=args.device,
    unbounded=True
).to(args.device)
actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
net_c1 = Net(
    args.state_shape,
    args.action_shape,
    hidden_sizes=args.hidden_sizes,
    concat=True,
    device=args.device
)
critic1 = Critic(net_c1, device=args.device).to(args.device)
critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
net_c2 = Net(
    args.state_shape,
    args.action_shape,
    hidden_sizes=args.hidden_sizes,
    concat=True,
    device=args.device
)
critic2 = Critic(net_c2, device=args.device).to(args.device)
critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

if args.auto_alpha:
    target_entropy = -np.prod(env.action_space.shape)
    log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
    alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
    args.alpha = (target_entropy, log_alpha, alpha_optim)

policy = SACPolicy(
    actor,
    actor_optim,
    critic1,
    critic1_optim,
    critic2,
    critic2_optim,
    tau=args.tau,
    gamma=args.gamma,
    alpha=args.alpha,
    reward_normalization=args.rew_norm,
    estimation_step=args.n_step,
    action_space=env.action_space
)
```

此处为对模型的定义，通过`tianshou`中的`Net()`来表示神经网络模型，它继承`torch.nn.Module`或者`tf.keras.Model`。输入为探针获得的速度值或者压力值由之前定义，输入口为状态值的shape即探针的个数。对于本框架中，使用的SAC算法，tianshou中给出的框架为两个Q网络以及一个策略网络，Q网络用于输入状态和动作，输出两个Q值(Q值用来判断该状态和动作是否好，并且可以根据Q值来对策略网络进行更新)，为了防止过高估计Q值，导致策略优化不稳定。这是一种类似于TD3算法的技巧，即使用两个独立的Q网络，**每次更新时取两个Q值中的最小值作为目标值**，这样可以减少Q函数的方差，提高策略的性能；策略网络用于输入状态输出随机动作。

在语句中就是由以下结构组成SAC模型

```
actor网络
actor_optim
critic1网络
critic1_optim
critic2网络
critic2_optim
```

collector的构成

```
# collector
train_collector = AsyncCollector(
    policy,
    train_envs,
    VectorReplayBuffer(args.buffer_size, len(train_envs)),
    exploration_noise=True
)
test_collector=None
```

`AsyncCollector()`是tianshou内的一个类，用于异步收集数据，即在多个环境中并行地执行策略，并将转移存储到缓冲区内具体内容如下

- ```
  - policy: 用于执行动作的策略网络。
  - train_envs: 用于训练的环境列表。
  - VectorReplayBuffer: 用于存储转移的缓冲区。
  - exploration_noise: 是否在动作上添加探索噪声。
  ```

如果要使用最好的policy进行测试则需要对test_collector进行设置。

```
	# log_name =
	log_path = os.path.join(args.logdir, args.task, 'sac')
    # logger
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == 'wandb':
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            update_interval=10,
            config=args,
            project=args.wandb_project,
        )
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer,update_interval=1,save_interval=args.save_interval,)
    else:  # wandb
        logger.load(writer)
```

首先创建存储的路径为`log/openfoam-v0/sac`

将`log_path`内的数据写入到TensorBoard内，以便可视化和分析，并且把args内的参数都写入`writer`，此后将根据不同的日志记录器保存或加载数据。

    def save_best_fn(policy):
        torch.save(
                    {
                        'model': policy.state_dict(),
                        # 'optim': optim.state_dict(),
                    }, os.path.join(log_path, 'best_model.pth')
                   )
    
    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold
    
    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save(
            {
                'model': policy.state_dict(),
                # 'optim': optim.state_dict(),
            }, os.path.join(log_path, 'checkpoint.pth')
        )
        # pickle.dump(
        #     train_collector.buffer,
        #     open(os.path.join(log_path, 'train_buffer.pkl'), "wb")
        # )

###### save_best_fn()函数

利用将policy.state_dict()将模型内每一层与其对应的参数张量建立映射关系，将可以训练的层保存至模型中，路径为log_path/best_model.pth

###### stop_fn()函数

配合程序末尾347行语句：`assert stop_fn(result['best_reward'])`用来设置训练停止返回的值，此处设置为最大奖励()大于等于设置的奖励门槛。

###### save_checkpoint_fn()函数

每个epoch，每隔一个env_step就进行保存。

```
def train_fn(epoch, env_step):
    # eps annnealing, just a demo
    if env_step <= 10000:
        policy.set_eps(args.eps_train)
    elif env_step <= 50000:
        eps = args.eps_train - (env_step - 10000) / \
            40000 * (0.9 * args.eps_train)
        policy.set_eps(eps)
    else:
        policy.set_eps(0.1 * args.eps_train)

def test_fn(epoch, env_step):
    policy.set_eps(args.eps_test)

if args.resume:
    # load from existing checkpoint
    print(f"Loading agent under {log_path}")
    ckpt_path = os.path.join(log_path, 'checkpoint.pth')
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        policy.load_state_dict(checkpoint['model'])
        # policy.optim.load_state_dict(checkpoint['optim'])
        print("Successfully restore policy and optim.")
    else:
        print("Fail to restore policy and optim.")
    buffer_path = os.path.join(log_path, 'train_buffer.pkl')
    if os.path.exists(buffer_path):
        train_collector.buffer = pickle.load(open(buffer_path, "rb"))
        print("Successfully restore buffer.")
    else:
        print("Fail to restore buffer.")

print(policy.training,policy.updating)
# trainer
result = offpolicy_trainer(
    policy,
    train_collector,
    test_collector,
    args.epoch,
    args.step_per_epoch,
    args.step_per_collect,
    args.test_num,
    args.batch_size,
    update_per_step=args.update_per_step,
    stop_fn=stop_fn,
    save_best_fn=save_best_fn,
    save_checkpoint_fn=save_checkpoint_fn,
    logger=logger,
    resume_from_log=args.resume,
)
assert stop_fn(result['best_reward'])
print("anything done")
```

###### train_fn函数

用来设置策略的探索率，根据env-step来调整探索率，分为了三个等级：10000、50000、以上。

###### test_fn函数

就以测试的参数为主，因为其不需要更新参数。

###### if args.resume

通过if语句判断是否需要从checkpoint出发，继续进行训练。

语句中还包含了是否恢复buffer_path，在该框架内并没有保存缓冲区的语句。

###### trainer构成

使用`offpolicy_trainer`用于训练基于离线策略的算法，接受参数，例如policy，collector…，返回一个字典，包含训练过程中的一些统计信息，如平均奖励、平均长度、平均损失等。



### cfd.py文件

是对调用cfd过程进行定义，其主要在environment_tianshou.py中被调用

1. run对训练过程中的数值模拟部分进行定义
2. run_init进行分块并且运行到初始时间步

##### run函数

该函数主要被应用于openfoam，对几何模型进行分块，并运行，完成后进行后处理整合

```python
def run(path, foam_params, writeInterval,deltaT, start_time, end_time):
    # 对被控制的位置进行定义：`control_dict_path`为每个环境中`'/system/controlDict'`
    control_dict_path = path + '/system/controlDict'
    # 判断end_time的值是否属于int或float类型
    assert isinstance(end_time, (int, float)), f'TypeError: end_time must be int or float type'
	# 以可读写形式打开control_dict_path文件即controlDict文件
    with open(control_dict_path, 'r+') as f:
        # 将其中内容都写入content变量中
        content = f.read()
        # ，并判断start_time是否为latestTime类型或是为startTime类型，并将对应的类型写入到startFrom后
        if start_time == 'latestTime':
            content = re.sub(f'(startFrom\s+).*;', f'\g<1>latestTime;', content)
        elif isinstance(start_time, (int, float)):
            content = re.sub(f'(startFrom\s+).*;', f'\g<1>startTime;', content)
            content = re.sub(f'(startTime\s+).+;', f'\g<1>{start_time};', content)
        else:
            assert False, f'TypeError: start_time must be int, float or specific strings type'
        # 对controlDict内的其他内容进行更改
        content = re.sub(f'(endTime\s+).*;', f'\g<1>{end_time};', content)
        content = re.sub(f'(writeInterval\s+).*;', f'\g<1>{writeInterval};', content)
        content = re.sub(f'(deltaT\s+).*;', f'\g<1>{deltaT};', content)
        # 寻找controlDict文件的开头并且将文件变为空值，再将content内的内容写入文件内，实现对文件内内容的修改
        f.seek(0)
        f.truncate()
        f.write(content)
	# 对是否需要进行记录进行设定
    if foam_params['verbose']:
        subprocess.run(
            f'cd {path}'  + ' && ' + 'decomposePar -force',
            shell=True, check=True, executable='/bin/bash'
        )
        mpi_process = subprocess.Popen(
            f'cd {path}' + ' && ' + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel',
            shell=True, executable='/bin/bash'
        )

        mpi_process.communicate()
        subprocess.run(
            f'cd {path}'  + ' && ' + 'reconstructPar',
            shell=True, check=True, executable='/bin/bash'
        )
    else:
        subprocess.run(
            f'cd {path}'  + ' && ' + 'decomposePar -force > /dev/null',
            shell=True, check=True, executable='/bin/bash'
        )
        mpi_process = subprocess.Popen(
            f'cd {path}'  + ' && ' + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel > /dev/null',
            shell=True, executable='/bin/bash'
        )

        mpi_process.communicate()
        subprocess.run(
            f'cd {path}' + ' && ' + 'reconstructPar > /dev/null',
            shell=True, check=True, executable='/bin/bash'
        )

```

该函数内包含`re.sub()`

###### re.sub

其中正则表达式内参数 `\d` (匹配任何数字), `\w` (匹配任何字母数字字符), 以及 `\s` (匹配任何空格字符).  `.` (匹配任何除换行符以外的字符), `^` (匹配任何字符串的开头), and `$` (匹配任何字符串结尾)，\ 该反斜杠字符表示转义字符，用于匹配特殊字符，`?`表示匹配前面字符零或一次，`*`表示匹配前面的字符任意次数。

##### 含有装饰器@utils.timeit('OpenFOAM_init')的run_init函数

```python
@utils.timeit('OpenFOAM_init')
def run_init(path, foam_params):
    """Run simulation in initial period.
    # 将模拟初始化
	# 函数内参数类型以及含义如下：
    Parameters
    ----------
    path : str
        Path to simulation file.
    foam_params : list
        Some parameters set before.

    Examples
    --------
    from DRLinFluids.cfd import run_init
    """
    # 判断初始时间步是否已经存在，存在则报错
    assert foam_params['cfd_init_time'], f'\n\nInitialization before training is compulsory!\n'
    control_dict_path = path + '/system/controlDict'
    decompose_par_dict_path = path + '/system/decomposeParDict'
	# 控制模拟进行分块并行计算
    # 若要使用其他分块方法，此处也需要相应改变
    with open(decompose_par_dict_path, 'r+') as f:
        content = f.read()
        content = re.sub(f'(numberOfSubdomains\s+)\d+;', f'\g<1>{foam_params["num_processor"]};', content)
        f.seek(0)
        f.truncate()
        f.write(content)
	# 作用同run函数，以0开始计算算到'cfd_init_time'停止
    with open(control_dict_path, 'r+') as f:
        content = f.read()
        content = re.sub(f'(application\s+).+;', f'\g<1>{foam_params["solver"]};', content)
        content = re.sub(f'(deltaT\s+).*;', f'\g<1>{foam_params["delta_t"]};', content)
        content = re.sub(f'(startFrom\s+).*;', f'\g<1>startTime;', content)
        content = re.sub(f'(startTime\s+).+;', f'\g<1>0;', content)
        content = re.sub(f'(endTime\s+).+;', f'\g<1>{foam_params["cfd_init_time"]};', content)
        content = re.sub(f'(writeInterval\s+).+;', f'\g<1>{foam_params["cfd_init_time"]};', content)
        content = re.sub(f'(purgeWrite\s+).+;', f'\g<1>0;', content)
        f.seek(0)
        f.truncate()
        f.write(content)

    if foam_params['verbose']:
        subprocess.run(
            f'cd {path}' + ' && ' + 'decomposePar -force',
            shell=True, check=True, executable='/bin/bash'
        )
        mpi_process = subprocess.Popen(
            f'cd {path}' + ' && ' + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel',
            shell=True, executable='/bin/bash'
        )

        mpi_process.communicate()
        subprocess.run(
            f'cd {path}' + ' && ' + 'reconstructPar',
            shell=True, check=True, executable='/bin/bash'
        )
    else:
        subprocess.run(
            f'cd {path}'  + ' && ' + 'decomposePar -force > /dev/null',
            shell=True, check=True, executable='/bin/bash'
        )
        mpi_process = subprocess.Popen(
            f'cd {path}'  + ' && ' + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel > /dev/null',
            shell=True, executable='/bin/bash'
        )

        mpi_process.communicate()
        subprocess.run(
            f'cd {path}' + ' && ' + 'reconstructPar > /dev/null',
            shell=True, check=True, executable='/bin/bash'
        )
```

该处引用于utils.py文件内的timeit作为装饰器，该装饰器作用是用于计算函数运行时间。

该函数作用用于计算每个环境内从0时刻到`cfd_init_time`所用的时间。

### utils.py文件

1. 读取文件内内容，并且重新拼接，被用作提取力系数。
2. 定义计时函数，用来统计模拟运行的时间。
3. 读取不同文件内容，并且分别把他们放入变量中，并决定是否需要写为csv文件
4. 定义函数，两个函数相互配合，被用作将Agent输出的动作，寻找每个环境内的最大时间，更改射流值。

#### resultant_force函数

```python
def resultant_force(dataframe, saver=False):
    # 将dataframe文件第一列所有元素都填入Time变量中 
    Time = dataframe.iloc[:, 0]
    # 将1-3列元素提取装入Fp，并附抬头，FX,FY,FZ。Fv，Fo同理
    Fp = dataframe.iloc[:, [1, 2, 3]]
    Fp.columns = ['FX', 'FY', 'FZ']
    Fv = dataframe.iloc[:, [4, 5, 6]]
    Fv.columns = ['FX', 'FY', 'FZ']
    Fo = dataframe.iloc[:, [7, 8, 9]]
    Fo.columns = ['FX', 'FY', 'FZ']
    # Mp，Mv，Mo同上述
    Mp = dataframe.iloc[:, [10, 11, 12]]
    Mp.columns = ['MX', 'MY', 'MZ']
    Mv = dataframe.iloc[:, [13, 14, 15]]
    Mv.columns = ['MX', 'MY', 'MZ']
    Mo = dataframe.iloc[:, [16, 17, 18]]
    Mo.columns = ['MX', 'MY', 'MZ']
	# 上述中变量pvo在分别为压力、粘度、流体多孔之间的空隙
    # 将上述变量按列，排列拼接
    result = pd.concat([pd.concat([Time, Fp + Fv + Fo], axis=1), Mp + Mv + Mo], axis=1)
	# 若saver为True则保存为csv文件
    if saver:
        result.to_csv(saver)

    return result
```

#### timeit函数

在后续会被作为装饰器引入用来记录所使用的时间

```
def timeit(params):
    """Record the running time of the function, params passes in the display string"""
    def inner(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func(*args, **kwargs)
            end_time = time.time()
            print(f'{params} running time：', np.around(end_time-start_time, decimals=2), 's')
        return wrapper
    return inner
```

#### read_foam_file函数

用来阅读文件内数据，包含探针内数据、后处理中力参数等等，并且可以设置每列数据的抬头，返回该数据

```python
def read_foam_file(path, mandatory=False, saver=False, dimension=3):
    """Extract any file including system/probes, postProcessing/.*,
    store results in a DataFrame object with column title,
    and without specify index.

    Parameters
    ----------
    path : str
        Path to simulation file.
    mandatory : bool, optional
    saver : bool, optional
    dimension : int
        Running dimension.

    Returns
    -------
    data_frame_obj
        DataFrame object.

    Examples
    --------
    from DRLinFluids.utils import read_foam_file

    note
    --------
    When reading the forces file, the header is defined as:
    Fp: sum of forces induced by pressure
    Fv: sum of forces induced by viscous
    Fo: sum of forces induced by porous
    Mp: sum of moments induced by pressure
    Mv: sum of moments induced by viscous
    Mo: sum of moments induced by porous
    """
    # Determine whether to read the system/probe file，若读system文件则需读取probes
    # 整个if判断语句用于判断不同程序，读取文件构成不同的data_frame_obj
    # 小tips探针文件不存在后缀
    if path.split('/')[-2] == 'system':
        if path.split('/')[-1] == 'probes':
            with open(path, 'r') as f:
                content_total = f.read()
                # 将content_total内的所有（以及之前的空格删掉，之后再将）；删掉，赋值给right_str
                right_str = re.sub('\);?', '', re.sub('[ \t]*\(', '', content_total))
                annotation_num = 0
            # 应用for循环切割right_str变量，并且获得以数字开头的行数即annotation_num
            for line in right_str.split('\n'):
                if re.search('^-?\d+', line):
                    break
                annotation_num += 1
            # 将right_str字符串转换为文件形式，并将其探针值给data_frame_obj中，并赋值x,y,z
            right_content = StringIO(right_str)
            data_frame_obj = pd.read_csv(right_content, sep=' ', skiprows=annotation_num, header=None,
                                         names=['x', 'y', 'z'])
        else:
            # 如果这个路径不是probes则报错
            data_frame_obj = False
            assert data_frame_obj, f'Unknown system/file type\n{path}'
    # Determine whether to read postProcessing/* files
    # 确定是否阅读后处理文件
    elif path.split('/')[-4] == 'postProcessing':
        # Write the postProcess file to the variable content_total and count the number of comment lines annotation_num
        # 确定数据在第几行
        with open(path, 'r') as f:
            content_total = f.read()
            f.seek(0)
            content_lines = f.readlines()
            annotation_num = 0
            for line in content_lines:
                if line[0] == '#':
                    annotation_num += 1
                else:
                    break
        # 如果是读取force文件，去除forces内括号以及制表符、#
        if path.split('/')[-1] == 'forces.dat':
            column_name = ['Time']
            column_name.extend(['Fpx', 'Fpy', 'Fpz'])
            column_name.extend(['Fvx', 'Fvy', 'Fvz'])
            column_name.extend(['Fox', 'Foy', 'Foz'])
            column_name.extend(['Mpx', 'Mpy', 'Mpz'])
            column_name.extend(['Mvx', 'Mvy', 'Mvz'])
            column_name.extend(['Mox', 'Moy', 'Moz'])
            right_content = StringIO(re.sub('\)', '', re.sub('\(', '', re.sub('\t+', '\t', re.sub(' +', '\t',
                                                                                                  re.sub('# ', '',
                                                                                                         re.sub(
                                                                                                             '[ \t]+\n',
                                                                                                             '\n',
                                                                                                             content_total)))))))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None, index_col=False,
                                         names=column_name)
        elif path.split('/')[-1] == 'p':
            right_content = StringIO(
                re.sub('\t\n', '\n', re.sub(' +', '\t', re.sub('# ', '', re.sub('[ \t]+\n', '\n', content_total)))))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num - 1, index_col=False)
        elif path.split('/')[-1] == 'U':
            column_name = ['Time']
            for n in range(annotation_num - 1):
                column_name.append(f'Ux_{n}')
                column_name.append(f'Uy_{n}')
                column_name.append(f'Uz_{n}')
            # print(len(column_name))
            right_content = StringIO(
                re.sub(' +', '\t', re.sub('[\(\)]', '', re.sub('# ', '', re.sub('[ \t]+\n', '\n', content_total)))))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None, index_col=False,
                                         names=column_name)
            if dimension == 2:
                drop_column = [i for i in column_name if re.search('^Uz_\d', i)]
                data_frame_obj.drop(drop_column, axis=1, inplace=True)
        elif path.split('/')[-1] == 'forceCoeffs.dat':
            column_name = ['Time', 'Cm', 'Cd', 'Cl', 'Cl(f)', 'Cl(r)']
            right_content = StringIO(re.sub('[ \t]+', '\t', re.sub('[ \t]+\n', '\n', content_total)))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None, index_col=False,
                                         names=column_name)
        # If they do not match, return an error directly
        else:
            if mandatory:
                right_content = StringIO(re.sub(' ', '', re.sub('# ', '', content_total)))
                data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None)
            else:
                data_frame_obj = -1
                assert 0, f'Unknown file type, you can force function to read it by using \'mandatory\' parameters (scalar-like data structure)\n{path}'
    else:
        data_frame_obj = -1
        assert 0, f'Unknown folder path\n{path}'
	# 如果saver为True则保存为csv文件
    if saver:
        data_frame_obj.to_csv(saver, index=False, header=False)

    return data_frame_obj
```

#### actions2dict()函数

将agent输出的动作输入到变量(x,y,z)中

```python
def actions2dict(entry_dict, reinforcement_learning_var, agent_actions):
    """Write parameters to velocity or pressure files in OpenFOAM.
    # 将参数写入到OpenFOAM的速度、压力文件中
	# 在environments_tianshou.py中引用方式是entry_dict参数为agent_foam参数中的entry_dict_q0，reinforcement_learning_var参数为agent_foam参数中的variables_q0，agent_actions为start_actions
    Parameters
    ----------
    entry_dict : list
        list for change item，需要改变的项列表
    reinforcement_learning_var : tuple
        reinforcement learning variables，包含了强化学习的变量
    agent_actions : tuple
        The action value generated by the agent.其是由代理生成的动作值

    Returns
    -------
    actions_dict
        Specify reinforcement learning variables.

    Examples
    --------
    from DRLinFluids.utils import actions2dict
    reinforcement_learning_var_example = (x, y, z)
    agent_actions_example = (1, 2, 3)

    note
    --------
    # 在动作定义中，只能由简单的加减乘除不可以包含正则表达式
    In particular, entry_dict should not contain regular expression related,
    minmax_value: Specify independent variables and expressions related to boundary conditions.
    Note that only the most basic calculation expressions are accepted, and operation functions
    in math or numpy cannot be used temporarily.
    entry_example = {
        'U': {
            'JET1': '({x} 0 0)',
            'JET2': '(0 {y} 0)',
            'JET3': '(0 0 {z})',
            'JET4': '{x+y+z}',
            'JET(4|5)': '{x+y+z}'  # X cannot contain regular expressions and must be separated manually
        },
        'k': {
            'JET1': '({0.5*x} 0 0)',
            'JET2': '(0 {0.5*y*y} 0)',
            'JET3': '(0 0 {2*z**0.5})',
            'JET4': '{x+2*y+3*z}'
        }
    }
    """
    # 将两个元组中的值一一对应，并且打包成一个字典，字典的key是强化学习参数，值是智能体的动作
    mapping_dict = dict(zip(reinforcement_learning_var, agent_actions))
	# 将entry_dict字典转换为json字符串并且用四个空格进行缩进，转换为json易于程序或应用之间的数据交换
    entry_str_org = json.dumps(entry_dict, indent=4)
    entry_str_temp = re.sub(r'{\n', r'{{\n', entry_str_org)
    entry_str_temp = re.sub(r'}\n', r'}}\n', entry_str_temp)
    # 匹配以}结尾的字符串，并将其用}}替代}
    entry_str_temp = re.sub(r'}$', r'}}', entry_str_temp)
    entry_str_temp = re.sub(r'},', r'}},', entry_str_temp)
    # eval()函数将字符串作为python代码执行，将其中的x，y，t用值代替，例如'q1': '{8}',
    entry_str_final = eval(f'f"""{entry_str_temp}"""', mapping_dict)
    # 将json字符串转换为python字典，actions_dict为
    actions_dict = json.loads(entry_str_final)
	# 返回被赋值的动作形式
    return actions_dict
```

#### dict2foam()函数

匹配每个时刻的U，并将其内部的jet中的q值进行修改

```python
def dict2foam(flow_var_directory, actions_dict):
    """Write parameters to velocity or pressure files in OpenFOAM.
	# flow_var_directory参数一般用当前算例中最大时间路径表示，actions_dict用上述函数中所返回的actions_dict表示，当前所输出的力
    Parameters
    ----------
    flow_var_directory : str
        Path to simulation file.
    actions_dict : tuple
        Specify reinforcement learning variables

    Examples
    --------
    from DRLinFluids.utils import dict2foam
    """
    # 对actions_dict内所有的键值对进行遍历
    for flow_var, flow_dict in actions_dict.items():
        with open('/'.join([flow_var_directory, flow_var]), 'r+') as f:
        # open用来打开每个时刻U文件
		#         'U': {{
        #     'JET': {{
        #         'q1': '{y}',
        #     }}
        # }}
            content = f.read()
            for entry, value_dict in flow_dict.items():
            # flow_dict为
            # 'JET': {{
            #     'q1': '{y}',
            # }}
                for keyword, value in value_dict.items():
                    # value_dict为
                    # 'q1': '{y}'
                    # re.sub在U文件内，匹配JET内的q0或q1或t把其中的内容进行修改
                    content = re.sub(f'({entry}(?:\n.*?)*{keyword}\s+).*;', f'\g<1>{value};', content)
                # content = re.sub(f'({entry}(?:\n.*?)*value\s+uniform\s+).*;', f'\g<1>{value};', content)
            f.seek(0)
            f.truncate()
            f.write(content)
```

#### get_current_time_path函数

用来获取每个环境中当前时间下、最大时间文件夹名称与路径

```python
def get_current_time_path(path):
    """Enter the root directory of the OpenFOAM study, the return value is
    (current (maximum) time folder name|absolute path).
	# 接收一个路径作为参数，返回当前或最大时间文件夹名称或绝对路径。
	# 此处的path指的是每个env(i)
    Parameters
    ----------
    path : str
        Path to simulation file.

    Returns
    -------
    start_time_str
        Current latest time folder name.
    start_time_path
        Current latest time folder path.

    Examples
    --------
    from DRLinFluids.utils import get_current_time_path
    """
    time_list = [i for i in os.listdir(path)]
    temp_list = time_list
    # 应用enumerate()函数，获取其内的索引与值，索引和值为同一数值，将除数字以外的文件路径设置为-1
    for i, value in enumerate(time_list):
        if re.search('^\d+\.?\d*', value):
            pass
        else:
            temp_list[i] = -1
    # 遍历剩余的索引值，以float类型存在time_list_to_num中，
    time_list_to_num = [np.float(i) for i in temp_list]
    start_time_str = time_list[np.argmax(time_list_to_num)]
    start_time_path = '/'.join([path, start_time_str])

    return start_time_str, start_time_path
```

### environments_tianshou.py文件

1. 该文件是文章的主题文件，用于创建训练环境以及包含许多重要逻辑以及一些参数的定义。
1. 定义轨迹过程，并记录相关数据
1. 定义奖励函数
1. 记录每个episode内的数据，并进行reset
1. 对openfoam计算得到的升力阻力进行smooth


#### 类OpenFoam_tianshou()

```
class OpenFoam_tianshou(gym.Env):
   """The main DRLinFluids Gym class.
    It encapsulates an environment with arbitrary behind-the-scenes dynamics.
    An environment can be partially or fully observed.

    Parameters
    ----------
    foam_root_path : str
        Path to simulation file.
    foam_params : list
        simulation parameters.
    agent_params : list
        DRL parameters.
    state_params : list
        Running dimension.

    Examples
    --------
    from DRLinFluids.environments_tianshou import OpenFoam_tianshou

    note
    --------
    The main API methods that users of this class need to know are:
    - :meth:`step` - Takes a step in the environment using an action returning the next observation, reward,
    if the environment terminated and more information.
    - :meth:`reward_function` - Define reward_funtion and calculate.
    - :meth:`reset` - Resets the environment to an initial state, returning the initial observation.
    """
```

##### 变量、参数的设置

```python
def __init__(self, foam_root_path, foam_params, agent_params, state_params, server=True):
   super().__init__()
   self.foam_params = foam_params
   self.agent_params = agent_params
   self.state_params = state_params
   self.foam_root_path = foam_root_path
   # Automatically read probes files and extract information
   self.state_params['probe_info'] = utils.read_foam_file(
      '/'.join([foam_root_path, 'system', 'probes'])
   )
   # Record a series of variables for each trajectory and pass to plotly
   self.dashboard_data = {}
   # Record each Trajectory start timestamp
   self.trajectory_start_time = 0
   # Record each Trajectory end timestamp
   self.trajectory_end_time = 0
   # Record the number of episodes
   self.num_episode = 0
   # Record all step information
   self.info_list = []
   # Record the reward for each episode
   self.episode_reward_sequence = []
   # Used to pass the data generated during the step() process
   self.exec_info = {}
   # Initialize the trajectory counter
   self.num_trajectory = 0
   # Initialize each step (trajectory) reward
   self.trajectory_reward = np.array([])
   # Initially record all trajectory rewards for all episodes
   self.all_episode_trajectory_reward = pd.DataFrame()
   # Record the State list of all Trajectory in the current Episode
   self.state_data = np.array([])
   # Initialize cumulative rewards
   self.episode_reward = 0
   # Initialize the action sequence in the episode
   self.actions_sequence = np.array([])
   # Action at the start of each step
   self.start_actions = 0
   # Action at the end of each step
   self.end_actions = 0
   # Record actions in steps
   self.single_step_actions = np.array([])
   # Initialize a pandas object to record all actions of all episodes
   self.all_episode_actions = pd.DataFrame()
   # Initialize a pandas object to record all actual output actions of all episodes
   self.all_episode_decorated_actions = pd.DataFrame()
   # Initialize a pandas object to record all actual output actions of all episodes
   self.all_episode_single_step_actions = pd.DataFrame()
   # Initialize the velocity(state) file
   self.probe_velocity_df = pd.DataFrame()
   # Initialize the pressure(state) file
   self.probe_pressure_df = pd.DataFrame()
   # Initialize forces(reward) in the current time step
   self.force_df = pd.DataFrame()
   # Initialize the current time step internal force coefficient forceCoeffs(reward)
   self.force_Coeffs_df = pd.DataFrame()
   # Initialize full cycle force forces(reward)
   self.history_force_df = pd.DataFrame()
   # Initialize the full cycle force coefficient forceCoeffs(reward)
   self.history_force_Coeffs_df = pd.DataFrame()
   # Read the initial flow field moment of the input and adjust its format
   self.cfd_init_time_str = str(float(foam_params['cfd_init_time'])).rstrip('0').rstrip('.')
   # Avoid errors in hex conversion, resulting in inability to read the file correctly
   self.decimal = int(np.max([
      len(str(agent_params['interaction_period']).split('.')[-1]),
      len(str(foam_params['cfd_init_time']).split('.')[-1])
   ]))
```

```python
		# 该处相当于定义server的含义，server=True则对程序进行初始化计时。
    	if server:
			# Initialize a flow field, otherwise the initial state is empty and learning cannot be performed
			cfd.run_init(foam_root_path, foam_params)
			# Store the initialized flow field result (state) in the _init file to avoid repeated reading and writing
            # 储存初始化流场的数据将其存储与pressure_table_init中，避免重复读写
			self.velocity_table_init = utils.read_foam_file(
				foam_root_path + f'/postProcessing/probes/0/U',
				dimension=self.foam_params['num_dimension']
			)
			self.pressure_table_init = utils.read_foam_file(
				foam_root_path + f'/postProcessing/probes/0/p',
				dimension=self.foam_params['num_dimension']
			)
```

对状态、动作空间进行定义

`action_space=spaces.Box()`其中各个参数的意义如下：

先确定动作空间维度，即`shape(m, n)`代表有m个动作，每个动作有n个元素，例如shape(3, 2),动作空间即设置为[[1, 2], [3, 4], [5, 6]]。这里的batch size是3，因为有3个动作。

在确定动作空间的维度后，我们需要确定最低最高值，若m为3时，我们可以这样定义最大最小值`low=np.array([0, -1]), high=np.array([1, 1]),`

此处举一个m为3，n为2的动作空间设置：

`self.action_space = spaces.Box(low=np.array([0, -1， 4]), high=np.array([2, 3， 40]), shape(3, ), dtype=np.float32)`

```python
		# 对状态空间进行定义
		# 用压力值作为状态空间内的状态值
		if self.state_params['type'] == 'pressure':
            # 将状态空间用Box定义，无上下限，空间的维度以探针的个数为行，且空间内值的类型为float
			self.state_space = spaces.Box(low=-np.Inf, high=np.Inf,
			                              shape=(int(self.state_params['probe_info'].shape[0]),), dtype=np.float32)
		# using velocity as a training parameter
        # 二维速度空间的维度为2倍的探针值，速度分为u、v
		elif self.state_params['type'] == 'velocity':
			if self.foam_params['num_dimension'] == 2:
				self.state_space = spaces.Box(low=-np.Inf, high=np.Inf,
				                              shape=(2 * int(self.state_params['probe_info'].shape[0]),),
				                              dtype=np.float32)
			elif self.foam_params['num_dimension'] == 3:
				self.state_space = spaces.Box(low=-np.Inf, high=np.Inf,
				                              shape=(3 * int(self.state_params['probe_info'].shape[0]),),
				                              dtype=np.float32)
			else:
				assert 0, 'Simulation type error'
		else:
			assert 0, 'No define state type error'

		# action_space
        # 对动作空间进行定义，agent_params中的最大最小值定义作为该空间内的上下限，动作空间的维度为agent_params中variables_q0的长度，类型同样为float32
		self.action_space = spaces.Box(self.agent_params['minmax_value'][0], self.agent_params['minmax_value'][1],
		                               shape=(len(self.agent_params['variables_q0']),), dtype=np.float32)
		self.seed()
		self.viewer = None
```

##### step函数

记录了一个轨迹的过程：

{

记录开始时间

轨迹+1

把当前动作加入到存储动作的列表中，

将当前动作输入至openfoam文件最大时刻的U中

运行openfoam，起始时间和终止时间相隔一个interaction_period

将获得的数据存储到不同的变量中

并且把算得的状态值作为下一个trajectory的初始状态

计算reward值

记录结束时间

}

```python
# Takes an action and returns a tuple, running a timestep of the environment's dynamics,
# and resetting the state of the environment at the end
# 在环境中执行一个动作，并且返回一个元组，运行cfd过程，并且在最后重新设置环境状态
def step(self,actions: np.ndarray):
   """Run one timestep of the environment's dynamics."""
   # 以当前时间戳作为开始时间
   self.trajectory_start_time = time()
   # 从1开始计算轨迹
   self.num_trajectory += 1
   if actions is None:
      print("carefully, no action given; by default, no jet!")

   self.actions_sequence = np.append(self.actions_sequence, actions)
	# 判断轨迹数为多少，如果为1则开始动作为0，结束动作为当前列表内的动作
    # 如果大于1则开始动作为列表中倒数第二个动作(上一个动作)，结束动作为列表内最后一个动作(当前动作)
   if self.num_trajectory < 1.5:
      self.start_actions = [0]
      self.end_actions = [self.actions_sequence[0]]
   else:
      self.start_actions = [self.actions_sequence[-2]]
      self.end_actions = [self.actions_sequence[-1]]
	# 应用np.around函数将数组或矩阵中的元素四舍五入到指定小数位，以此方法来算的一个trajectory的时间
   start_time_float = np.around(
      float(self.cfd_init_time_str) + (self.num_trajectory - 1) * self.agent_params['interaction_period'],
      decimals=self.decimal
   )
   end_time_float = np.around(start_time_float + self.agent_params['interaction_period'], decimals=self.decimal)

   # Find the current latest time folder, as startTime, to specify the action write folder path
   # 寻找当前最新的文件夹，并将其作为初始时间，并且把其名称与路径分别赋值给start_time_filenam、start_time_path,并将指定的动作写入指定文件中(U),具体函数是在utils.py中进行定义
   start_time_filename, start_time_path = utils.get_current_time_path(self.foam_root_path)

   # Change the start_action issued by the agent to the corresponding time folder
   utils.dict2foam(
      start_time_path,
      utils.actions2dict(self.agent_params['entry_dict_q0'], self.agent_params['variables_q0'],
                         self.start_actions)
   )

   # Change the end_action issued by the agent to the corresponding time folder
   utils.dict2foam(
      start_time_path,
      utils.actions2dict(self.agent_params['entry_dict_q1'], self.agent_params['variables_q1'], self.end_actions)
   )

   start_time = [start_time_float]
   # Change the t0 issued by the agent to the corresponding time folder
   utils.dict2foam(
      start_time_path,
      utils.actions2dict(self.agent_params['entry_dict_t0'], self.agent_params['variables_t0'], start_time)
   )
	# 记录当前时间戳作为模拟开始时间
   simulation_start_time = time()
	# 调用cfd.py内的run函数进行
   cfd.run(
      self.foam_root_path,
      self.foam_params,
      self.agent_params['writeInterval'],
      self.agent_params['deltaT'],
      start_time_float, end_time_float
   )
	# 记录当前时间戳作为模拟结束时间
   simulation_end_time = time()
	# 阅读探针所获取的状态值（速度），并返回一个变量，该变量含有文件内的所有数据
   # Read velocity(state) file
   self.probe_velocity_df = utils.read_foam_file(
      self.foam_root_path + f'/postProcessing/probes/{start_time_filename}/U',
      dimension=self.foam_params['num_dimension']
   )
	# 阅读探针所获取的状态值（压力），并返回一个变量，该变量含有文件内的所有数据
   # Read pressure file (state)
   self.probe_pressure_df = utils.read_foam_file(
      self.foam_root_path + f'/postProcessing/probes/{start_time_filename}/p',
      dimension=self.foam_params['num_dimension']
   )
	# 阅读当前最大时间文件内的力数据，并返回一个变量，该变量含有文件内的所有数据
   # Read the forces.dat file and output it directly in the form of total force (reward)
   self.force_df = utils.resultant_force(
      utils.read_foam_file(
         self.foam_root_path + f'/postProcessing/forcesIncompressible/{start_time_filename}/forces.dat'
      )
   )
	# 阅读当前最大时间文件内的力系数，并返回一个变量，该变量含有文件内的所有数据
   # Read the force coefficient forceCoeffs.dat file (reward)
   self.force_Coeffs_df = utils.read_foam_file(
      self.foam_root_path + f'/postProcessing/forceCoeffsIncompressible/{start_time_filename}/forceCoeffs.dat'
   )
	# 将所有force_df,force_Coeffs_ds都储存在history_force_df中
   # Links all full cycle historical force and force coefficient data prior to the current trajectory
	# .reset()函数表示重置数据帧或系列的索引，并且参数drop决定是否需要保留原索引（True表示不保留原索引）
   if self.num_trajectory < 1.5:
      self.history_force_df = self.force_df
      self.history_force_Coeffs_df = self.force_Coeffs_df
   else:
      self.history_force_df = pd.concat([self.history_force_df, self.force_df[1:]]).reset_index(drop=True)
      self.history_force_Coeffs_df = pd.concat(
         [self.history_force_Coeffs_df, self.force_Coeffs_df[1:]]
      ).reset_index(drop=True)

   # Use the last line of the result file as the next state
	# 使用当前探针获得的状态值，并将最后一行的所有元素作为下一个状态的初始值
   if self.state_params['type'] == 'pressure':
      next_state = self.probe_pressure_df.iloc[-1, 1:].to_numpy()
   elif self.state_params['type'] == 'velocity':
      next_state = self.probe_velocity_df.iloc[-1, 1:].to_numpy()
   else:
      next_state = False
      assert next_state, 'No define state type'
    # 并将数据记录入state_data内，state_data用来记录整个episode的状态值
   self.state_data = np.append(self.state_data, next_state)
	# 调用奖励函数奖励，计算reward值
   # Calculate the reward value
   reward = self.reward_function()
    # 每一个trajectory，都输出（trajectory数、上一个动作、当前动作、奖励值）
   print(self.num_trajectory, self.start_actions, self.end_actions,reward)
   # Record the reward value of each trajectory and episode
    # 记录每个trajectory的奖励，并且把他们记录在trajectory_reward内
   self.trajectory_reward = np.append(self.trajectory_reward, reward)
   self.episode_reward += reward

   # Termination condition
	# 终端情况
   terminal = False
	# 将当前时间戳作为轨迹的结束时间，一个轨迹的整个过程就是
   self.trajectory_end_time = time()
	# 将下列参数值输入至列表中，并将该列表添加到info_list中
   # Used to pass the data generated during the step() process
   self.exec_info = {
      'episode': self.num_episode,
      'trajectory': self.num_trajectory,
      'start_time_float': start_time_float,
      'end_time_float': end_time_float,
      'timestampStart': self.trajectory_start_time,
      'timestampEnd': self.trajectory_end_time,
      'current_trajectory_reward': reward,
      'episode_reward': self.episode_reward,
      'actions': actions,
      'cfd_running_time': simulation_end_time - simulation_start_time,
      'number_cfd_timestep': int(np.around((end_time_float - start_time_float) / self.foam_params['delta_t'])),
      'envName': self.foam_root_path.split('/')[-1],
      'current_state': self.state_data[-2],
      'next_state': next_state,
   }
   self.info_list.append(self.exec_info)
	# 返回下个状态、奖励以及端口
   return next_state, reward, terminal, {}
```

##### 带有装饰器的reward_funtion()函数

```python
@abstractmethod
def reward_function(self):
   """Define reward and formulate it as an abstract method, which means that
   when instantiating the OpenFoam class, the reward_function function must be overridden"""
   vortex_shedding_period =self.agent_params['interaction_period']
   drug_coeffs_sliding_average = self.force_coeffs_sliding_average(vortex_shedding_period)[0]
   lift_coeffs_sliding_average = self.force_coeffs_sliding_average(vortex_shedding_period)[1]
   print(0 - (drug_coeffs_sliding_average - 0.286), 0.2 * np.abs(lift_coeffs_sliding_average - 0.88516))
   return 0 - (drug_coeffs_sliding_average - 0.286) - 0.2 * np.abs(lift_coeffs_sliding_average - 0.88516)
```

作用同`envobject_airfoil.py`内的`reward_function()`

带有装饰器`@abstractmethod`，这是python类中的抽象方法，只有被调用才会被实例化。

##### reset函数

每个episode完成后，恢复到最初的状态(openfoam中设置的cfd_init_time)

```python
	def reset(self):
		"""Resets the environment to an initial state and returns the initial observation."""
        # 若episode为0，则创建record文件夹，若大于0则将episode_reward值填入变量episode_reward_sequence中，并将他们写为csv'文件内,应用.to_csv()语句若存储路径一致则会将语句前的变量替换掉原csv文件内的内容，如果仅仅是添加至内容末尾，则可以在语句内添加参数，mode='a',即.to_csv(路径, mode='a')
		if self.num_episode < 0.5:
			os.makedirs(self.foam_root_path + '/record')
		else:
			# Extract the optimal action in the entire episode, skip the first initialization process
			self.episode_reward_sequence.append(self.episode_reward)
			pd.DataFrame(
				self.episode_reward_sequence
			).to_csv(self.foam_root_path + '/record/total_reward.csv', index=False, header=False)
            # 如果当前的奖励值是最大的奖励值则将当前记录episode内的action输入至best_actions.csv文件中，并且把当前episode内的力系数存到best_history_force_Coeffs_df.csv中，并且在info.txt内将当前的数值进行修改
			if self.episode_reward_sequence[-1] == np.max(self.episode_reward_sequence):
				pd.DataFrame(
					self.actions_sequence
				).to_csv(self.foam_root_path + '/record/best_actions.csv', index=False, header=False)
				pd.DataFrame(
					self.history_force_Coeffs_df
				).to_csv(self.foam_root_path + '/record/best_history_force_Coeffs_df.csv', index=False, header=False)
				with open(self.foam_root_path + '/record/info.txt', 'w') as f:
					f.write(f'Current number of best reward episode is {self.num_episode}')
		# Output all actions up to the current trajectory
        # 将该episode内的所有动作与奖励值分别放入all_episode_actions，all_episode_trajectory_reward中，其为列表。
		if self.num_episode == 1:
			self.all_episode_actions = pd.DataFrame(self.actions_sequence)
			self.all_episode_trajectory_reward = pd.DataFrame(self.trajectory_reward)
		else:
			self.all_episode_actions[self.num_episode - 1] = pd.DataFrame(self.actions_sequence)
			self.all_episode_trajectory_reward[self.num_episode - 1] = pd.DataFrame(self.trajectory_reward)
		# 将上述两个变量，以及history_force_Coeffs_df变量保存为csv文件于record内
		# Save action, reward and lift-drag information
		self.all_episode_actions.to_csv(
			self.foam_root_path + '/record/all_episode_actions.csv', index=False, header=False
		)
		self.all_episode_trajectory_reward.to_csv(
			self.foam_root_path + '/record/all_episode_trajectory_reward.csv', index=False, header=False
		)
		self.history_force_Coeffs_df.to_csv(
			self.foam_root_path + f'/record/history_force_Coeffs_df_{self.num_episode}.csv', index=False, header=False
		)
		self.info_list = pd.DataFrame(self.info_list)
		self.info_list.to_csv(
			self.foam_root_path + f'/record/info_list_{self.num_episode}.csv'
		)
        # episode值增加1，并将其他相关变量重新设置
		# update episode
		self.num_episode += 1
		# Reset the trajectory in an episode
		self.num_trajectory = 0
		# Reset cumulative reward
		self.episode_reward = 0
		# Reset the action sequence in the episode
		self.actions_sequence = []
		# Reset the reward sequence in the trajectory
		self.trajectory_reward = []
		# Reset the actual action sequence of actions
		self.decorated_actions_sequence = []
		# Reset historical force and force coefficients
		self.history_force_df = pd.DataFrame()
		self.history_force_Coeffs_df = pd.DataFrame()
		self.info_list = []
		# TODO The flow field needs to be initialized, delete all files, but do not need to
		#  reset the control dictionary, because the set calculation time is already included in the step() function
		# Delete all time folders except 0 and the initial flow field folder and the postprocessing folder postProcessing
        # 识别env中除去0以外的其他数字文件，同时删除postProcessing文件
		for f_name in os.listdir(self.foam_root_path):
			if re.search(r'^\d+\.?\d*', f_name):
				if (f_name != '0') and (f_name != self.cfd_init_time_str):
					shutil.rmtree('/'.join([self.foam_root_path, f_name]))
			elif f_name == 'postProcessing':
				shutil.rmtree('/'.join([self.foam_root_path, f_name]))
			else:
				pass
		# 新episode的初始状态为cfd_init_time时，探针获取的压力与速度值。
		if self.state_params['type'] == 'pressure':
			init_state = self.pressure_table_init.iloc[-1, 1:].to_numpy()
		elif self.state_params['type'] == 'velocity':
			init_state = self.velocity_table_init.iloc[-1, 1:].to_numpy()
		else:
			init_state = False
			assert init_state, 'No define state type'
		# 并将状态数据以及初始状态放入至state_data中
		# Put the initial state into the state array
		self.state_data = np.append(self.state_data, init_state)

		return init_state

```

##### force_coeffs_sliding_average()函数

```python
def force_coeffs_sliding_average(self, sliding_time_interval):
   # Calculate the number of sampling time points
   sampling_num = int(sliding_time_interval / self.foam_params['delta_t'])
   # Calculate the sliding average of the lift coefficient over a vortex shedding cycle
   if self.history_force_Coeffs_df.shape[0] <= sampling_num:
      sliding_average_cd = np.mean(signal.savgol_filter(self.history_force_Coeffs_df.iloc[:, 2], 25, 0))
      sliding_average_cl = np.mean(signal.savgol_filter(self.history_force_Coeffs_df.iloc[:, 3], 25, 0))
   else:
      sliding_average_cd = np.mean(
         signal.savgol_filter(self.history_force_Coeffs_df.iloc[-sampling_num:, 2], 25, 0))
      sliding_average_cl = np.mean(
         signal.savgol_filter(self.history_force_Coeffs_df.iloc[-sampling_num:, 3], 25, 0))
   return sliding_average_cd, sliding_average_cl
```

该函数获取由OpenFOAM计算得到的升力、阻力值，通过函数将其smooth，并计算其平均值，该平均值即为奖励函数中所用到的升力阻力值。

### test_policy.py

## 特别注意、其他知识：

### 关于python函数、语句、数据类型问题：

程序中不同的数据类型，不同的函数无法共同应用其中：

1. `pd.DataFrame()`是建立一个二维的、可大可小、有标签轴(行和列)的表格数据结构

2. `utils.py`中`actions2dict`函数中存在语句`dict(zip())`他将variables定义的值与接收智能体给出的动作的值进行配对，组成字典，注意agent_actions即动作值，其的形式必须为可迭代形式，在此处设置为'list'
3. 注意元组形式，例如：   'y': ('x2'),和'y': ('x1','x2',)同为一维shape，因此定义动作空间需要重新设置参考



### 关于OpenFOAM内设置问题：

时间步设置方式：时间步长的计算公式是由库郎数来决定的，其描述的是时间步长和空间步长之间的关系。计算公式如下：
$$
deltaT=Co*min(dx/umax,dy/vmax,dz/wmax)
$$
其中dx,dy,dz分别为网格在x,y,z方向上的长度，umax,vmax,wmax,分别为速度场在x,y,z方向上的最大值



## 修改示例

### 1.修改多输出（多射流）内容，该版本为jet123都是被控制量

#### launch文件：

主要修改test_sac_with_il()函数，将其中内容修改为如下

```python
# 此处设置多个entry是因为，他们都需要在U内进行修改，但都放入U的value内该变量属于一维数组，在之后调用以及替换不方便
entry_dict_q0 = {
    'U': {
        'JET1': {
            'q0': '{x1}',
        },
    }
}

entry_dict_q1 = {
    'U': {
        'JET2': {
            'q0': '{x2}',
        },
    }
}

entry_dict_q2 = {
    'U': {
        'JET3': {
            'q0': '{x3}',
        },
    }
}

entry_dict_q3 = {
    'U': {
        'JET1': {
            'q1': '{y1}',
        },
    }
}

entry_dict_q4 = {
    'U': {
        'JET2': {
            'q1': '{y2}',
        },
    }
}

entry_dict_q5 = {
    'U': {
        'JET3': {
            'q1': '{y3}',
        },
    }
}
entry_dict_t0 = {
    'U': {
        'JET1': {
            't0': '{t}'
        },
        'JET2': {
            't0': '{t}'
        },
        'JET3': {
            't0': '{t}'
        },
    }
}

agent_params = {
    'entry_dict_q0': entry_dict_q0,
    'entry_dict_q1': entry_dict_q1,
    'entry_dict_q2': entry_dict_q2,
    'entry_dict_q3': entry_dict_q3,
    'entry_dict_q4': entry_dict_q4,
    'entry_dict_q5': entry_dict_q5,
    'entry_dict_t0': entry_dict_t0,
    'deltaA': 0.01,
    'minmax_value': (-5, 5),
    'interaction_period': 0.005,
    'purgeWrite_numbers': 0,
    'writeInterval': 0.005,
    'deltaT': 0.0001,
    # 定义entry_dict_q0的同时，也要定义variables_q0
    'variables_q0': ('x1',),
    'variables_q1': ('x2',),
    'variables_q2': ('x3',),
    'variables_q3': ('y1',),
    'variables_q4': ('y2',),
    'variables_q5': ('y3',),
    'variables_t0': ('t',),
    'verbose': False,
    "zero_net_Qs": True,
}
```

修改完对应参数，需要在`environment_tianshou.py`文件内对step函数进行修改

#### environment_tianshou.py文件

设置承载action的变量，start_action为U中的q0，end_action为U中的q1

```python
# Action at the start of each step
self.start_actions_q0 = 0
self.start_actions_q1 = 0
self.start_actions_q2 = 0
# Action at the end of each step
self.end_actions_q3 = 0
self.end_actions_q4 = 0
self.end_actions_q5 = 0
```

因为输出动作为三个不同的值，因此需要将动作空间定义为三维，即shape=(3,)

```python
# action_space
self.action_space = spaces.Box(self.agent_params['minmax_value'][0], self.agent_params['minmax_value'][1],
                               shape=(3,), dtype=np.float32)
self.seed()
self.viewer = None
```

##### step函数

```python
# 将动作值赋值给变量，以便后续使用（动作存储在self.actions_sequence）
if self.num_trajectory < 1.5:
    self.start_actions_q0 = [0]
    self.start_actions_q1 = [0]
    self.start_actions_q2 = [0]
    self.end_actions_q3 = [self.actions_sequence[0]]
    self.end_actions_q4 = [self.actions_sequence[1]]
    self.end_actions_q5 = [self.actions_sequence[2]]
    # 此处为测试不同变量所输出值以及类型所用到的语句可忽略
    # print('start_actions为', self.start_actions_q0)
    # print('end_actions为', self.end_actions_q3)
    # print('actions_sequence为', self.actions_sequence)
    # print('action_type', type(self.actions_sequence))
    # print('start_action_type', type(self.start_actions_q0))
    # print('end_action_type', type(self.end_actions_q3))
else:
    self.start_actions_q0 = [self.actions_sequence[-6]]
    self.start_actions_q1 = [self.actions_sequence[-5]]
    self.start_actions_q2 = [self.actions_sequence[-4]]
    self.end_actions_q3 = [self.actions_sequence[-3]]
    self.end_actions_q4 = [self.actions_sequence[-2]]
    self.end_actions_q5 = [self.actions_sequence[-1]]
```

相应调用utils文件的语句中的参数也需要进行改变（具体参数以及函数作用在上文中有详细解释）

```python
# Change the start_action issued by the agent to the corresponding time folder
utils.dict2foam(
    start_time_path,
    utils.actions2dict(self.agent_params['entry_dict_q0'], self.agent_params['variables_q0'], self.start_actions_q0)
)

# Change the start_action issued by the agent to the corresponding time folder
utils.dict2foam(
    start_time_path,
    utils.actions2dict(self.agent_params['entry_dict_q1'], self.agent_params['variables_q1'], self.start_actions_q1)
)

# Change the start_action issued by the agent to the corresponding time folder
utils.dict2foam(
    start_time_path,
    utils.actions2dict(self.agent_params['entry_dict_q2'], self.agent_params['variables_q2'], self.start_actions_q2)
)

# Change the end_action issued by the agent to the corresponding time folder
utils.dict2foam(
    start_time_path,
    utils.actions2dict(self.agent_params['entry_dict_q3'], self.agent_params['variables_q3'], self.end_actions_q3)
)

# Change the end_action issued by the agent to the corresponding time folder
utils.dict2foam(
    start_time_path,
    utils.actions2dict(self.agent_params['entry_dict_q4'], self.agent_params['variables_q4'], self.end_actions_q4)
)

# 同样语句，将U内值替换
utils.dict2foam(
    start_time_path,
    utils.actions2dict(self.agent_params['entry_dict_q5'], self.agent_params['variables_q5'], self.end_actions_q5)
)
```

修改的同时要注意数据类型，使用list类型赋值给start或end，此处如果有更好的方法可提出。

### 2.修改多输出（多射流），该版本为使射流为0质量射流

#### launch文件：

此处不进行修改

#### environment_tianshou.py文件：

```python
# action_space
# 首先就是动作空间的定义，由于神经网络实际的输出动作就只有两个值，因此将动作空间设定为(2,)
self.action_space = spaces.Box(self.agent_params['minmax_value'][0], self.agent_params['minmax_value'][1],
                               shape=(2,), dtype=np.float32)
self.seed()
self.viewer = None
```

step函数

```python
def step(self, actions: np.ndarray):
    """Run one timestep of the environment's dynamics."""
    self.trajectory_start_time = time()
    self.num_trajectory += 1
    if actions is None:
        print("carefully, no action given; by default, no jet!")
    # actions_sequence的形式为numpy数组
    self.actions_sequence = np.append(self.actions_sequence, actions)

    if self.num_trajectory < 1.5:
        # 动作空间设置为2，因此在actions_sequence中每两个值为一个trajectory所记录的动作值
        self.start_actions_q0 = [0]
        self.start_actions_q1 = [0]
        # self.start_actions_q2 = [0]
        self.end_actions_q3 = [self.actions_sequence[0]]
        self.end_actions_q4 = [self.actions_sequence[1]]
        # self.end_actions_q5 = [self.actions_sequence[2]]
    else:
        self.start_actions_q0 = [self.actions_sequence[-4]]
        self.start_actions_q1 = [self.actions_sequence[-3]]
        # self.start_actions_q2 = [self.actions_sequence[-4]]
        self.end_actions_q3 = [self.actions_sequence[-2]]
        self.end_actions_q4 = [self.actions_sequence[-1]]
        # self.end_actions_q5 = [self.actions_sequence[-1]]

    start_time_float = np.around(
        float(self.cfd_init_time_str) + (self.num_trajectory - 1) * self.agent_params['interaction_period'],
        decimals=self.decimal
    )
    end_time_float = np.around(start_time_float + self.agent_params['interaction_period'], decimals=self.decimal)

    # Find the current latest time folder, as startTime, to specify the action write folder path
    start_time_filename, start_time_path = utils.get_current_time_path(self.foam_root_path)

    # Change the start_action issued by the agent to the corresponding time folder
    utils.dict2foam(
        start_time_path,
        utils.actions2dict(self.agent_params['entry_dict_q0'], self.agent_params['variables_q0'], self.start_actions_q0)
    )

    # Change the start_action issued by the agent to the corresponding time folder
    utils.dict2foam(
        start_time_path,
        utils.actions2dict(self.agent_params['entry_dict_q1'], self.agent_params['variables_q1'], self.start_actions_q1)
    )

    # Change the start_action issued by the agent to the corresponding time folder
    # 此处即将jet3的q0值定义为jet12的q0值相加的相反数
    self.start_actions_q2 = [0 - x - y for x, y in zip(self.start_actions_q0, self.start_actions_q1)]
    utils.dict2foam(
        start_time_path,
        utils.actions2dict(self.agent_params['entry_dict_q2'], self.agent_params['variables_q2'], self.start_actions_q2)
    )

    # Change the end_action issued by the agent to the corresponding time folder
    utils.dict2foam(
        start_time_path,
        utils.actions2dict(self.agent_params['entry_dict_q3'], self.agent_params['variables_q3'], self.end_actions_q3)
    )

    # Change the end_action issued by the agent to the corresponding time folder
    utils.dict2foam(
        start_time_path,
        utils.actions2dict(self.agent_params['entry_dict_q4'], self.agent_params['variables_q4'], self.end_actions_q4)
    )

    # 同样语句，将U内值替换
    # 此处同样，jet3的q1值定义为jet12的q1值相加的相反数
    self.end_actions_q5 = [0 - z - w for z, w in zip(self.end_actions_q4, self.end_actions_q3)]
    utils.dict2foam(
        start_time_path,
        utils.actions2dict(self.agent_params['entry_dict_q5'], self.agent_params['variables_q5'], self.end_actions_q5)
    )
```

### 3.本地化应用（举例说明openfoamv2006版本，大致思路可以参考）

如果只要在openfoamv8下就可实现想要的功能，就无需更改下列内容。

主要集中于程序部分，openfoam部分就不做介绍

#### launch文件：

`get_args`部分，参数值不进行改变。

test_sac_with_il部分：

```python
def test_sac_with_il(args=get_args()):
    # define parameters
    # Make environments:
    # you can also try with SubprocVectorEnv
    # 由于不同的openfoam版本算例不同，他们的分块方法也不同,该例子所采用的是simple分块，为了配合cfd.py文件，此处在foam_params内添加参数
    foam_params = {
        'delta_t': 0.0001,
        'solver': 'pimpleFoam',
        'num_processor': 5,
        'decompose_xyz':'(5 1 1)',
        'of_env_init': 'source ~/OpenFOAM/OpenFOAM-v2006/etc/bashrc',
        'cfd_init_time': 0.001,  # 初始化流场，初始化state,更加明显的对比
        'num_dimension': 2,
        'verbose': False,
        'probes_number':1,
    }
    entry_dict_q0 = {
        'U': {
            'JET': {
                'amplitude': '{x}',
            },
        }
    }

    agent_params = {
        'entry_dict_q0': entry_dict_q0,
        # 'entry_dict_q1': entry_dict_q1,
        # 'entry_dict_q2': entry_dict_q2,
        # 'entry_dict_q3': entry_dict_q3,
        # 'entry_dict_q4': entry_dict_q4,
        # 'entry_dict_q5': entry_dict_q5,
        # 'entry_dict_t0': entry_dict_t0,
        'deltaA': 0.006,
        'minmax_value': (-5, 5),
        'interaction_period': 0.003,
        'purgeWrite_numbers': 0,
        'writeInterval': 0.003,
        'deltaT': 0.0001,
        'variables_q0': ('x',),
        # 'variables_q1': ('x2',),
        # 'variables_q2': ('x3',),
        # 'variables_q3': ('y1',),
        # 'variables_q4': ('y2',),
        # 'variables_q5': ('y3',),
        # 'variables_t0': ('t',),
        'verbose': False,
        "zero_net_Qs": True,
        'control_number':1,
    }
    state_params = {
        'type': 'velocity'
    }
    root_path = os.getcwd()
    env_name_list = sorted([envs for envs in os.listdir(root_path) if re.search(r'^env\d+$', envs)])
    env_path_list = ['/'.join([root_path, i]) for i in env_name_list]

    env = envobject_airfoil.FlowAroundAirfoil2D(
        foam_root_path=env_path_list[0],
        foam_params=foam_params,
        agent_params=agent_params,
        state_params=state_params,
    )
    train_envs = SubprocVectorEnv(
        [lambda x=i: gym.make(args.task,foam_root_path=x,
                              foam_params=foam_params,
                              agent_params=agent_params,
                              state_params=state_params,
                              ) for i in env_path_list[0:args.training_num]],
        wait_num=args.training_num, timeout=0.2
    )
    # # test_envs = gym.make(args.task)
    # test_envs = SubprocVectorEnv(
    #     [lambda x=i: gym.make(args.task,foam_root_path=x,
    #                           foam_params=foam_params,
    #                           agent_params=agent_params,
    #                           state_params=state_params,
    #                           size=x, sleep=x) for i in env_path_list[args.training_num:(args.training_num+args.test_num)]]
    # )
    test_envs=None
    print(env.state_space.shape,env.action_space.shape)
    args.state_shape = env.state_space.shape or env.state_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    if args.reward_threshold is None:
        default_reward_threshold = {"OpenFoam-v0": 10, "Pendulum-v1": -250}
        args.reward_threshold = default_reward_threshold.get(
            args.task, env.spec.reward_threshold
        )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed([1, 2, 3, 4, 5])
    # test_envs.seed([5])

    # model
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net,
        args.action_shape,
        max_action=args.max_action,
        device=args.device,
        unbounded=True
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device
    )
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        reward_normalization=args.rew_norm,
        estimation_step=args.n_step,
        action_space=env.action_space
    )

    # collector
    train_collector = AsyncCollector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True
    )
    test_collector=None
    # train_collector.collect(n_step=args.buffer_size)

    # log
    # now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    # args.algo_name = "dqn_icm" if args.icm_lr_scale > 0 else "sac"
    # log_name =
    log_path = os.path.join(args.logdir, args.task, 'sac')
    # writer = SummaryWriter(log_path)
    # logger = TensorboardLogger(writer, save_interval=args.save_interval)
    # log_name = os.path.join(now)
    # log_path = os.path.join(args.logdir, log_name)

    # logger
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == 'wandb':
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            update_interval=10,
            config=args,
            project=args.wandb_project,
        )
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer,update_interval=1,save_interval=args.save_interval,)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):
        torch.save(
                    {
                        'model': policy.state_dict(),
                        # 'optim': optim.state_dict(),
                    }, os.path.join(log_path, 'best_model.pth')
                   )

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save(
            {
                'model': policy.state_dict(),
                # 'optim': optim.state_dict(),
            }, os.path.join(log_path, 'checkpoint.pth')
        )
        # pickle.dump(
        #     train_collector.buffer,
        #     open(os.path.join(log_path, 'train_buffer.pkl'), "wb")
        # )

    def train_fn(epoch, env_step):
        # eps annnealing, just a demo
        if env_step <= 10000:
            policy.set_eps(args.eps_train)
        elif env_step <= 50000:
            eps = args.eps_train - (env_step - 10000) / \
                40000 * (0.9 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * args.eps_train)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    if args.resume:
        # load from existing checkpoint
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, 'checkpoint.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            policy.load_state_dict(checkpoint['model'])
            # policy.optim.load_state_dict(checkpoint['optim'])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")
        buffer_path = os.path.join(log_path, 'train_buffer.pkl')
        if os.path.exists(buffer_path):
            train_collector.buffer = pickle.load(open(buffer_path, "rb"))
            print("Successfully restore buffer.")
        else:
            print("Fail to restore buffer.")

    print(policy.training,policy.updating)
    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        update_per_step=args.update_per_step,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
        resume_from_log=args.resume,
    )
    assert stop_fn(result['best_reward'])
    print("anything done")

    if __name__ == '__main__':
        pprint.pprint(result)
```

#### cfd.py文件

cfd文件内，主要针对分块部分进行修改

```python
def run_init(path, foam_params):
    assert foam_params['cfd_init_time'], f'\n\nInitialization before training is compulsory!\n'
    control_dict_path = path + '/system/controlDict'
    decompose_par_dict_path = path + '/system/decomposeParDict'
	
    # 此处即为该算例中分块的方法,此处需要匹配不同的分块方法，此处使用的是simple分块方法
    with open(decompose_par_dict_path, 'r+') as f:
        content = f.read()
        content = re.sub(f'(numberOfSubdomains\s+)\d+;', f'\g<1>{foam_params["num_processor"]};', content)
        content = re.sub(r'\s+n\s+\((\d+)\s+(\d+)\s+(\d+)\);', ' n ', content)
        content = re.sub(f'(\s+n\s+)', f'\g<1>{foam_params["decompose_xyz"]};', content)
        f.seek(0)
        f.truncate()
        f.write(content)

    with open(control_dict_path, 'r+') as f:
        content = f.read()
        content = re.sub(f'(application\s+).+;', f'\g<1>{foam_params["solver"]};', content)
        content = re.sub(f'(deltaT\s+).*;', f'\g<1>{foam_params["delta_t"]};', content)
        content = re.sub(f'(startFrom\s+).*;', f'\g<1>startTime;', content)
        content = re.sub(f'(startTime\s+).+;', f'\g<1>0;', content)
        content = re.sub(f'(endTime\s+).+;', f'\g<1>{foam_params["cfd_init_time"]};', content)
        content = re.sub(f'(writeInterval\s+).+;', f'\g<1>{foam_params["cfd_init_time"]};', content)
        content = re.sub(f'(purgeWrite\s+).+;', f'\g<1>0;', content)
        f.seek(0)
        f.truncate()
        f.write(content)

    if foam_params['verbose']:
        subprocess.run(
            f'cd {path}' + ' && ' + 'decomposePar -force',
            shell=True, check=True, executable='/bin/bash'
        )
        mpi_process = subprocess.Popen(
            f'cd {path}' + ' && ' + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel',
            shell=True, executable='/bin/bash'
        )

        mpi_process.communicate()
        subprocess.run(
            f'cd {path}' + ' && ' + 'reconstructPar',
            shell=True, check=True, executable='/bin/bash'
        )
    else:
        subprocess.run(
            f'cd {path}'  + ' && ' + 'decomposePar -force > /dev/null',
            shell=True, check=True, executable='/bin/bash'
        )
        mpi_process = subprocess.Popen(
            f'cd {path}'  + ' && ' + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel > /dev/null',
            shell=True, executable='/bin/bash'
        )

        mpi_process.communicate()
        subprocess.run(
            f'cd {path}' + ' && ' + 'reconstructPar > /dev/null',
            shell=True, check=True, executable='/bin/bash'
        )
```

#### utils.py文件

该文件是需要着重修改的位置，他是读取数据的重要文件

```python
def read_foam_file(path, mandatory=False, saver=False, dimension=3):
    # 此处并没有进行修改
    if path.split('/')[-2] == 'system':
        if path.split('/')[-1] == 'probes':
            with open(path, 'r') as f:
                content_total = f.read()
                right_str = re.sub('\);?', '', re.sub('[ \t]*\(', '', content_total))
                annotation_num = 0
            for line in right_str.split('\n'):
                if re.search('^-?\d+', line):
                    break
                annotation_num += 1
            right_content = StringIO(right_str)
            data_frame_obj = pd.read_csv(right_content, sep=' ', skiprows=annotation_num, header=None,
                                         names=['x', 'y', 'z'])
        else:
            data_frame_obj = False
            assert data_frame_obj, f'Unknown system/file type\n{path}'
    # Determine whether to read postProcessing/* files
    elif path.split('/')[-4] == 'postProcessing':
        # Write the postProcess file to the variable content_total and count the number of comment lines annotation_num
        with open(path, 'r') as f:
            content_total = f.read()
            f.seek(0)
            content_lines = f.readlines()
            annotation_num = 0
            total_content = []
            for line in content_lines:
                if line[0] == '#':
                    annotation_num += 1
                else:
                    break
        # 根据'force.dat'的数据类型，定义每列列明
        if path.split('/')[-1] == 'force.dat':
            column_name = ['Time']
            column_name.extend(['Fpx', 'Fpy', 'Fpz'])
            column_name.extend(['Fvx', 'Fvy', 'Fvz'])
            column_name.extend(['Fox', 'Foy', 'Foz'])
            # column_name.extend(['Mpx', 'Mpy', 'Mpz'])
            # column_name.extend(['Mvx', 'Mvy', 'Mvz'])
            # column_name.extend(['Mox', 'Moy', 'Moz'])
            # right_content = StringIO(re.sub('\)', '', re.sub('\(', '', re.sub('\t+', '\t', re.sub(' +', '\t',
            #                                                                                       re.sub('# ', '',
            #                                                                                              re.sub(
            #                                                                                                  '[ \t]+\n',
            #                                                                                                  '\n',
            #                                                                                                  content_total)))))))
            # column_name = ['Time', 'Cd', 'Cs', 'Cl', 'CmRoll', 'CmPitch', 'CmYaw', 'Cd(f)', 'Cd(r)', 'Cs(f)', 'Cs(r)',
            #                'Cl(f)', 'Cl(r)']
            right_content = StringIO(re.sub('[ \t]+', '\t', re.sub('[ \t]+\n', '\n', content_total)))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None, index_col=False,
                                         names=column_name)
        elif path.split('/')[-1] == 'p':
            column_name = ['Time']
            with open(path, 'r') as f:
                for content in f:
                    content = content.strip()
                    total_content.append(content)
                    new_content = '\n'.join(total_content)
                P_content = re.sub('\t\n', '\n',
                                   re.sub(' +', '\t', re.sub('#\s*', '', re.sub('[ \t]+\n', '\n', new_content))))
                for n in range(1):
                    column_name.append(f'{n}')
                right_content = StringIO(P_content)
                data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, index_col=False, names=column_name)
                # print(data_frame_obj)
        elif path.split('/')[-1] == 'U':
            column_name = ['Time']
            for n in range(annotation_num - 2):
                column_name.append(f'Ux_{n}')
                column_name.append(f'Uy_{n}')
                column_name.append(f'Uz_{n}')
            # print(len(column_name))
            with open(path, 'r') as f:
                for content in f:
                    content = content.strip()
                    total_content.append(content)
                    new_content = '\n'.join(total_content)
                U_content = re.sub(' +', '\t',
                                   re.sub('[\(\)]', '', re.sub('#\s*', '', re.sub('[ \t]+\n', '\n', new_content))))
                right_content = StringIO(U_content)
                data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None,
                                             index_col=False,
                                             names=column_name)
                if dimension == 2:
                    drop_column = [i for i in column_name if re.search('^Uz_\d', i)]
                    data_frame_obj.drop(drop_column, axis=1, inplace=True)
                # print(data_frame_obj)
        # elif path.split('/')[-1] == 'forceCoeffs.dat':
        #    column_name = ['Time', 'Cm', 'Cd', 'Cl', 'Cl(f)', 'Cl(r)']
        #     right_content = StringIO(re.sub('[ \t]+', '\t', re.sub('[ \t]+\n', '\n', content_total)))
        #     data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None, index_col=False,
        #                                  names=column_name)
        elif path.split('/')[-1] == 'coefficient.dat':
            column_name = ['Time', 'Cd', 'Cs', 'Cl', 'CmRoll', 'CmPitch', 'CmYaw', 'Cd(f)', 'Cd(r)', 'Cs(f)', 'Cs(r)', 'Cl(f)', 'Cl(r)']
            right_content = StringIO(re.sub('[ \t]+', '\t', re.sub('[ \t]+\n', '\n', content_total)))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None, index_col=False,
                                             names=column_name)
        # If they do not match, return an error directly
        else:
            if mandatory:
                right_content = StringIO(re.sub(' ', '', re.sub('# ', '', content_total)))
                data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None)
            else:
                data_frame_obj = -1
                assert 0, f'Unknown file type, you can force function to read it by using \'mandatory\' parameters (scalar-like data structure)\n{path}'
    else:
        data_frame_obj = -1
        assert 0, f'Unknown folder path\n{path}'

    if saver:
        data_frame_obj.to_csv(saver, index=False, header=False)

    return data_frame_obj
```

### 4.奖励函数创新

添加magnitude of vorticity到奖励函数中

#### openfoam内修改

openfoam中需要计算出vorticity的magnitude

添加mag文件(openfoam库中postProcessing文件内mag，在`mag文件`内的`fields`修改为`vorticity`，将mag填入`controlDict`文件内的`#includeFunc`)

将mag计算的在probes文件内显现出(在`probes文件`内的fields填入`vorticity`、`mag(vorticity)`)，在`postprocess`内`probes`内`0.0005文件`之后存有`mag(vorticity)`其可用于奖励函数内，作为奖惩的一部分，也要考虑其在奖励函数中的权重。

#### utils.py文件

首先要考虑如何将其读入程序中，作为数据储存，可以参考探针获取的P值

在`read_foam_file`函数中，添加以下内容

```python
elif path.split('/')[-1] == 'mag(vorticity)':
    right_content = StringIO(
        re.sub('\t\n', '\n', re.sub(' +', '\t', re.sub('# ', '', re.sub('[ \t]+\n', '\n', content_total)))))
    data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num - 1, index_col=False)
```

将vorticity内的内容读入到变量data_frame_obj中形式如下

```txt
       Time         0           1          2          3          4          5  \
0    0.0005  1.478296  101.113740  216.25246  229.26807  203.44277  176.92313   
1    0.0010  1.478296  101.113740  216.25246  229.26807  203.44277  176.92313   
2    0.0015  1.478296  101.113740  216.25246  229.26807  203.44277  176.92313   
3    0.0020  1.478296  101.113740  216.25246  229.26807  203.44277  176.92313   
4    0.0025  1.478296  101.113740  216.25246  229.26807  203.44277  176.92313   
..      ...       ...         ...        ...        ...        ...        ...   
256  0.1285  1.721797   96.298673  203.74510  214.45985  188.93641  163.02522   
257  0.1290  1.721797   96.298673  203.74510  214.45985  188.93641  163.02522   
258  0.1295  1.721797   96.298673  203.74510  214.45985  188.93641  163.02522   
259  0.1300  1.721797   96.298673  203.74510  214.45985  188.93641  163.02522   
260  0.1305  1.566179   96.669939  205.34379  217.18293  192.22109  166.50777 
```

#### env文件

参考force_coffeces的建立

创建变量部分（建议创建全局变量,注意每个变量的type，前两个为pd的数据类型，后两个为float数据类型

```python
# Initialize vorticity_magnitude
self.vorticity_magnitude = pd.DataFrame() # 记录一个trajectory从openfoam中提取的数据
self.history_vorticity_mag = pd.DataFrame() # 将每个trajectory获得的数据存入history中
self.average_vorticity_mag = 0 # 计算平均涡大小，应用于reward
self.max_vorticity_mag = 0 # 计算最大涡大小，应用于reward
```

要将数据从openfoam中读取出，考虑其位置，添加：

```python
# Read the vorticity magnitude file (reward)
self.vorticity_magnitude = utils.read_foam_file(
   self.foam_root_path + f'/postProcessing/probes/{vorticity_time}/mag(vorticity)'
current_vorticity = self.vorticity_magnitude.iloc[-1, 1:].to_numpy()
self.average_vorticity_mag = np.mean(current_vorticity)
self.max_vorticity_mag = np.amax(current_vorticity)
```

由于openfoam中探针存储mag(vorticity)文件的位置与探针存储P、U文件的位置不同，因此，需要创建`vorticity_time`变量

```python
# Find the current latest time folder, as startTime, to specify the action write folder path
start_time_filename, start_time_path = utils.get_current_time_path(self.foam_root_path)
x = str(self.agent_params['writeInterval'])
y = len(x)-1
z = float(start_time_filename)
z += self.foam_params['delta_t']
vorticity = round(z, y)
vorticity_time = str(vorticity) # 应用于记录时间
```

考虑到实际存储位置是当前最大时间值＋时间步，且放置float数据类型出现误差，因此采用round函数对其值进行保留位数，在相加之后将其转化为str类型

将vorticity_magnitude(即从probes内读取到的mag(voriticity)存入history_vorticityh_mag中)

```python
# Links all full cycle historical force and force coefficient data prior to the current trajectory
if self.num_trajectory < 1.5:
   self.history_force_df = self.force_df
   self.history_force_Coeffs_df = self.force_Coeffs_df
   self.history_vorticity_mag = self.vorticity_magnitude
else:
   self.history_force_df = pd.concat([self.history_force_df, self.force_df[1:]]).reset_index(drop=True)
   self.history_force_Coeffs_df = pd.concat(
      [self.history_force_Coeffs_df, self.force_Coeffs_df[1:]]
   ).reset_index(drop=True)
   self.history_vorticity_mag = pd.concat(
      [self.history_vorticity_mag, self.vorticity_magnitude[1:]]).reset_index(drop=True)
```

同时考虑可以将平均、最大涡记录到info中，因此：

```python
# Used to pass the data generated during the step() process
self.exec_info = {
   'episode': self.num_episode,
   'trajectory': self.num_trajectory,
   'start_time_float': start_time_float,
   'end_time_float': end_time_float,
   'timestampStart': self.trajectory_start_time,
   'timestampEnd': self.trajectory_end_time,
   'current_trajectory_reward': reward,
   'episode_reward': self.episode_reward,
   'actions': actions,
   'average_vorticity':self.average_vorticity_mag,
   'max_vorticity':self.max_vorticity_mag,
   'cfd_running_time': simulation_end_time - simulation_start_time,
   'number_cfd_timestep': int(np.around((end_time_float - start_time_float) / self.foam_params['delta_t'])),
   'envName': self.foam_root_path.split('/')[-1],
   'current_state': self.state_data[-2],
   'next_state': next_state,
}
```

将这些都处理完成则需要考虑到将其放入reward中其权重如何（考虑到不同变量之间的数量级

```python
def reward_function(self):
   """Define reward and formulate it as an abstract method, which means that
   when instantiating the OpenFoam class, the reward_function function must be overridden"""
   vortex_shedding_period = self.agent_params['interaction_period']
   drug_coeffs_sliding_average = self.force_coeffs_sliding_average(vortex_shedding_period)[0]
   lift_coeffs_sliding_average = self.force_coeffs_sliding_average(vortex_shedding_period)[1]
   max_voriticity_mag = self.max_vorticity_mag
   mean_vorticity_mag = self.average_vorticity_mag
   print(2.4 - drug_coeffs_sliding_average, 0.1 * np.abs(lift_coeffs_sliding_average),
        0.1 * (227-max_voriticity_mag), 0.1*(40-mean_vorticity_mag))
   return 2.4 - drug_coeffs_sliding_average - 0.1 * np.abs(lift_coeffs_sliding_average) - 0.1 * (227-max_voriticity_mag) - 0.1*(40-mean_vorticity_mag)
```

之后需要同history_force一样创建一个csv文件用来存储history_vorticity_mag，以及存储best_vorticity_mag

```python
    def reset(self):
      """Resets the environment to an initial state and returns the initial observation."""
      if self.num_episode < 0.5:
         os.makedirs(self.foam_root_path + '/record')
      else:
         # Extract the optimal action in the entire episode, skip the first initialization process
         self.episode_reward_sequence.append(self.episode_reward)
         pd.DataFrame(
            self.episode_reward_sequence
         ).to_csv(self.foam_root_path + '/record/total_reward.csv', index=False, header=False)
         if self.episode_reward_sequence[-1] == np.max(self.episode_reward_sequence):
            pd.DataFrame(
               self.actions_sequence
            ).to_csv(self.foam_root_path + '/record/best_actions.csv', index=False, header=False)
            pd.DataFrame(
               self.history_force_Coeffs_df
            ).to_csv(self.foam_root_path + '/record/best_history_force_Coeffs_df.csv', index=False, header=False)
            # 如果当前为最大reward值，则将vorticity存储为best.csv
            pd.DataFrame(
               self.history_vorticity_mag
            ).to_csv(self.foam_root_path + '/record/best_history_vorticity_magnitude.csv', index=False, header=False)
            with open(self.foam_root_path + '/record/info.txt', 'w') as f:
               f.write(f'Current number of best reward episode is {self.num_episode}')

      # Output all actions up to the current trajectory
      if self.num_episode == 1:
         self.all_episode_actions = pd.DataFrame(self.actions_sequence)
         self.all_episode_trajectory_reward = pd.DataFrame(self.trajectory_reward)
      else:
         self.all_episode_actions[self.num_episode - 1] = pd.DataFrame(self.actions_sequence)
         self.all_episode_trajectory_reward[self.num_episode - 1] = pd.DataFrame(self.trajectory_reward)

      # Save action, reward and lift-drag information
      self.all_episode_actions.to_csv(
         self.foam_root_path + '/record/all_episode_actions.csv', index=False, header=False
      )
      self.all_episode_trajectory_reward.to_csv(
         self.foam_root_path + '/record/all_episode_trajectory_reward.csv', index=False, header=False
      )
      self.history_force_Coeffs_df.to_csv(
         self.foam_root_path + f'/record/history_force_Coeffs_df_{self.num_episode}.csv', index=False, header=False
      )
        # 将其变化过程存储为history.csv
      self.history_vorticity_mag.to_csv(
         self.foam_root_path + f'/record/history_vorticity_mag_{self.num_episode}.csv', index=False, header=False
)
      self.info_list = pd.DataFrame(self.info_list)
      self.info_list.to_csv(
         self.foam_root_path + f'/record/info_list_{self.num_episode}.csv'
      )
      # update episode
      self.num_episode += 1
      # Reset the trajectory in an episode
      self.num_trajectory = 0
      # Reset cumulative reward
      self.episode_reward = 0
      # Reset the action sequence in the episode
      self.actions_sequence = []
      # Reset the reward sequence in the trajectory
      self.trajectory_reward = []
      # Reset the actual action sequence of actions
      self.decorated_actions_sequence = []
      # Reset historical force and force coefficients
      self.history_force_df = pd.DataFrame()
      self.history_force_Coeffs_df = pd.DataFrame()
      # 每个episode完成后都需要删除该episode存储下的vorticity_mag
      self.history_vorticity_mag = pd.DataFrame()
      self.info_list = []
```

## DMD拓展

### DMD程序阅读——根据nek5000程序

#### nek5000程序总体框架

DMD计算绘制存在于DMD.py文件（包含如下）

- 
