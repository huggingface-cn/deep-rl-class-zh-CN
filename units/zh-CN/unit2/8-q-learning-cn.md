# 初探Q-Learning

## 什么是Q-Learning？

Q-Learning是一种**离线策略的基于价值**的方法，它**使用时序差分方法来训练其动作-价值函数**：

- *离线策略*：我们将在本单元的最后讨论这个问题。
- *基于价值的方法*：通过训练一个价值函数或动作-价值函数来间接地找到最优策略，该函数能告诉我们**每个状态或每个状态-动作对的价值**。
- *使用时序差分方法*：**在每一步更新其动作-价值函数，而不是在回合结束时进行更新。**

**Q-Learning是我们用来训练 Q 函数的算法**，Q 函数是一个**动作-价值函数**，用于确定在特定状态下采取特定动作的价值。

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-function.jpg" alt="Q-function"/>
  <figcaption>给定一个状态和动作，Q函数将输出对应的状态-动作值(也叫做Q值)</figcaption>
</figure>
**Q来源于“Quality”（价值），即该状态下的动作价值。**

让我们回顾一下价值和奖励之间的区别：

- *状态的价值*或*状态-动作对的价值*是智能体在此状态（或状态-动作对）开始行动并按照其策略行事时预期的累积奖励。
- *奖励*是在状态下执行动作后**从环境中获得的反馈**。

在内部，Q函数有一个**Q-表，这个表中的每个单元格对应一个状态-动作对的价值。可以将这个Q-表视为Q函数的记忆或速查表。**

让我们通过一个迷宫的例子来解释以上内容。

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Maze-1.jpg" alt="Maze example"/>

我们对Q-表进行初始化，所以其中的值都为0. 这个表格**包含了每个状态的四个状态-动作值。**

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Maze-2.jpg" alt="Maze example"/>

在这里我们可以看到，**初始状态向上的状态-动作值为0：**

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Maze-3.jpg" alt="Maze example"/>

因此，Q函数包含一个Q-表，其中包含每个状态动作对的值。给定一个状态和动作，Q函数会在其Q-表中搜索并输出该值。

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-function-2.jpg" alt="Q-function"/>
</figure>


回顾一下，*Q-Learning*是一个包含以下过程的**强化学习算法**: 

- 训练一个*Q-函数*（一个**动作-价值函数**），其内部是一个**包含所有状态-动作对值的Q-表。**
- 给定状态和动作，我们的Q-函数**会在其Q-表中查找相应的值。**
- 当训练完成后，**我们有了一个最优的Q-函数，这意味着我们有了最优的Q-表。**
- 如果我们**拥有最优的Q-函数**，我们**拥有最优的策略**，因为我们**知道每个状态下应该采取的最佳动作。**

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/link-value-policy.jpg" alt="Link value policy"/>

但是，在开始时**我们的Q-表是没有用的，因为它给每个状态-动作对赋予了任意的值**（大多数情况下，我们把Q-表初始化为0）。随着智能体**探索环境并更新Q-表，它会给我们更好的近似最优策略。**

<figure class="image table text-center m-0 w-full">
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-1.jpg" alt="Q-learning"/>
  <figcaption>可以看到随着训练，Q-表变得更好了，因为借助它，我们可以知道每个状态-动作对的值。</figcaption>
</figure>
现在我们已经理解了什么是Q-Learning，Q-函数和Q-表，接下来**让我们深入了解一下Q-Learning算法**。

## Q-Learning 算法

这是Q-Learning 的伪代码；让我们研究一下其中的每一部分，并在实现它之前**用一个简单的例子看看它是如何工作的。**不要被它的形式吓到，它比看起来简单！

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-2.jpg" alt="Q-learning"/>

### 第一步: 初始化Q-表

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-3.jpg" alt="Q-learning"/>

我们需要初始化Q-表中的每个状态-动作值。**大多数情况下，我们用0来初始化。**

### 第二步: 使用epsilon贪心策略选择一个动作

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-4.jpg" alt="Q-learning"/>

epsilon-贪婪策略是一种处理探索与利用权衡的策略。

其思想是首先定义初始 epsilon ɛ = 1.0：

- *概率 1 — ɛ*：智能体进行**利用**（即智能体选择具有最高状态-动作对值的动作）。
- 概率 ɛ：**智能体进行探索**（尝试随机动作）。

在训练开始时，**由于 ɛ 值很高，进行探索的概率会很大，所以智能体大部分时间都在探索。但随着训练的进行，Q-表在估计中越来越准确，所以逐渐降低 epsilon 值**，因为智能体逐渐不再需要探索，而需要更多地进行利用。

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-5.jpg" alt="Q-learning"/>

### 第三步: 执行动作At, 得到奖励Rt+1和下一个状态St+1

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-6.jpg" alt="Q-learning"/>

### 第四步: 更新Q(St, At)

需要注意的是，在时序差分学习中，我们在与环境交互之后更新策略或价值函数（取决于我们选择的强化学习方法）。

为了计算时序差分目标(TD target)，**我们使用立即奖励 \(R_{t+1}\) 加上下一个状态最佳状态-动作对的折扣价值**（称之为自举）。

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-7.jpg" alt="Q-learning"/>

因此，(Q(S_t, A_t)\) **更新公式如下：**

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-8.jpg" alt="Q-learning"/>

这意味着要更新 \(Q(S_t, A_t)\)：

- 需要 \(S_t, A_t, R_{t+1}, S_{t+1}\)。
- 需要更新给定状态-动作对的Q值，使用TD target。

如何形成TD target？

1. 在采取动作后获得奖励 \(R_{t+1}\)。
2. 为了获得**最佳的下一个状态-动作对值**，使用贪婪策略来选择下一个最佳动作。需要注意的是，这不是一个 ε-贪婪策略，其将始终采取具有最高状态-动作值的动作。

然后，在此Q值更新完成后，将开始一个新的状态，并**再次使用 ε-贪婪策略选择动作。**

**这就是为什么我们说Q学习是一种离线策略算法。**

## 离线策略 vs 在线策略

它们之间只有细微的区别：

- *离线策略*：**在行动（推理）和更新（训练）部分中**使用**不同的策略**。

例如，使用Q学习，ε-贪婪策略（行动策略），与**用于选择最佳下一状态动作值来更新Q值（更新策略）**的贪婪策略不同。

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/off-on-1.jpg" alt="Off-on policy"/>
  <figcaption>行动策略</figcaption>
</figure>



与我们在训练部分中所使用的策略不同：

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/off-on-2.jpg" alt="Off-on policy"/>
  <figcaption>更新策略</figcaption>
</figure>



- 在线策略：在**行动和更新部分**中使用**相同的策略**。

例如，在另一种基于值的算法Sarsa中，**执行ε-贪心策略时，它选择的不是最优动作，而是下一个状态和相应的动作组合。**


<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/off-on-3.jpg" alt="Off-on policy"/>
    <figcaption>Sarsa</figcaption>
</figure>


<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/off-on-4.jpg" alt="Off-on policy"/>
</figure>
