# Q-Learning算法实例

为了更好的理解Q-Learning算法，我们举一个简单的例子：

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Maze-Example-2.jpg" alt="Maze-Example"/>

- 假如你是一个处在小迷宫中的小老鼠，并总是**从相同的起点开始出发**。
- 你的目标是**吃掉右下角的大块奶酪**，并且避免吃到毒药。毕竟相比于小块奶酪，谁会不喜欢大块奶酪呢？
- 在每次探索中，如果吃到毒药、**吃掉大块奶酪或行动超过五个步骤**，则该次探索结束。
- 学习率为0.1。
- Gamma（折扣率）为0.99。

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/q-ex-1.jpg" alt="Maze-Example"/>

奖励函数如下：

- **+0:** 去一个没有奶酪的状态。
- **+1:** 去一个有小奶酪的状态。
- **+10:** 去一个有一大堆奶酪的状态。
- **-10:** 去一个有毒药的状态，从而死亡。
- **+0** 如果我们花了超过五步。

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/q-ex-2.jpg" alt="Maze-Example"/>

**我们将使用 Q-Learning 算法**训练智能体，使其能够具有最优策略（即能够依次做出向右、向右、向下动作的策略）。

## 第一步: 初始化Q-table

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Example-1.jpg" alt="Maze-Example"/>

目前，**Q-table是没用的**；所以我们需要使用 Q-Learning 算法**来训练 Q 函数**。

我们进行 2 个训练时间步长的训练：

训练时间步长 1：

## 第二步：使用spsilon贪心策略选择动作

因为epsilon很大，等于1.0，所以你随机选择了一个向右的行动。

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/q-ex-3.jpg" alt="Maze-Example"/>

## 第三步：执行动作At，得到奖励Rt+1和新的状态St+1

向右走后，你得到了一块小奶酪，所以\(R_{t+1} = 1\)，并且你进入了一个新的状态。


<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/q-ex-4.jpg" alt="Maze-Example"/>

## 第四步：更新Q(St, At)

现在我们可以使用公式更新Q（St，At）。

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/q-ex-5.jpg" alt="Maze-Example"/>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Example-4.jpg" alt="Maze-Example"/>

第二次训练（不需要再对Q-table进行初始化）：

## 第二步：使用spsilon贪心策略选择动作

**由于epsilon还是很大，为0.99，所以你再次随机选择一个行动**（随着训练的进行，我们希望越来越少探索，所以我们把epsilon逐渐减小）。

你选择了一个向下的动作。**这是一个糟糕的行动，因为它让小老鼠吃到了毒药。**

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/q-ex-6.jpg" alt="Maze-Example"/>


## 第三步：执行动作At，得到奖励Rt+1和新的状态St+1

因为不小心吃到了毒药，所以**小老鼠不幸死亡**，**得到的奖励 Rt+1 = -10，。**

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/q-ex-7.jpg" alt="Maze-Example"/>

## 第四步：更新Q(St, At)

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/q-ex-8.jpg" alt="Maze-Example"/>

因为小老鼠牺牲了，所以我们开始了一个新的训练回合。但是我们可以看到，**在两个探索步骤后，智能体变得更聪明了。**

随着智能体继续探索和利用环境，并使用TD-target更新Q值，**Q表中的近似值越来越好。因此，在训练结束时，我们将获得Q函数的最优估计。**