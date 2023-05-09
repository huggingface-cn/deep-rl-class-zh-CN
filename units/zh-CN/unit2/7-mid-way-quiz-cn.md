# 学习进展测验

最好的学习方法是**对自己进行测试**。这能[避免过度自信](https://www.coursera.org/lecture/learning-how-to-learn/illusions-of-competence-BuFzf)并帮助我们找到需要加强的方面。


### Q1: 找到最优策略的两种主要方法是什么？

<Question
	choices={[
		{
			text: "基于策略的方法",
			explain: "使用基于策略的方法，我们直接训练策略来学习在给定状态下采取哪种行动。",
      correct: true
		},
		{
			text: "基于随机的方法",
			explain: ""
		},
    {
			text: "基于价值的方法",
			explain: "使用基于价值的方法，我们训练一个价值函数来学习哪些状态更有价值，并使用该价值函数来选择最优的动作。",
      correct: true
		},
		{
			text: "进化策略方法",
      explain: ""
		}
	]}
/>


### Q2: 什么是贝尔曼方程?

<details>
<summary>答案</summary>

**贝尔曼方程是一个递归方程**，其工作原理如下：我们不需要从每个状态的起点开始计算回报，而是可以将任意状态的价值视为:

Rt+1 + gamma * V(St+1)

立即奖励 + 之后状态的折扣价值。

</details>

### Q3: 贝尔曼方程的每个部分的定义是什么?

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/bellman4-quiz.jpg" alt="Bellman equation quiz"/>

<details>
<summary>答案</summary>


<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/bellman4.jpg" alt="Bellman equation solution"/>

</details>

### Q4: 蒙特卡洛（Monte Carlo）方法和时序差分（Temporal Difference, 简称TD）学习方法之间的差异是什么?

<Question
	choices={[
		{
			text: "在使用蒙特卡罗方法时,我们从一个完整的回合中更新价值函数",
			explain: "",
      correct: true
		},
    {
			text: "在使用蒙特卡罗方法时,我们从一个时间步中更新价值函数",
			explain: ""
		},
    {
			text: "在使用TD方法时,我们从一个完整的回合中更新价值函数",
			explain: ""
		},
    {
			text: "在使用TD方法时,我们从一个时间步中更新价值函数",
			explain: "",
      correct: true
		},
	]}
/>

### Q5: 时序差分算法中每一部分的定义是什么?

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/td-ex.jpg" alt="TD Learning exercise"/>

<details>
<summary>答案</summary>

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/TD-1.jpg" alt="TD Exercise"/>

</details>


### Q6: 蒙特卡洛算法中每一部分的定义是什么?

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/mc-ex.jpg" alt="MC Learning exercise"/>

<details>
<summary>答案</summary>

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/monte-carlo-approach.jpg" alt="MC Exercise"/>

</details>

恭喜你完成了这个测验🥳，如果你错过了一些要点，花点时间再读一遍前面的部分，以加强你的知识（😏）。