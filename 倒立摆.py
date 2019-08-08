# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

#载入库
import numpy as np
import tensorflow as tf
import gym

env = gym.make('CartPole-v0')
#创建CartPole问题的环境env

env.reset()
#初始化环境

random_episodes = 0
reward_sum = 0#奖励
while random_episodes < 10:
    env.render()#将CartPole问题的图像渲染出来

    observation, reward, done, _ = env.step(np.random.randint(0, 2))
    #使用np.random.randint(0, 2)产生随机的Action
    #然后使用env.step()执行随机的Action,并获取返回值
    #如果done标记为True,则表示这次试验结束，即倾角超过15度或者偏离中心过远导致任务失败

    reward_sum += reward
    if done:#如果试验结束
        random_episodes += 1
        print("game over,Reward for this episode was:", reward_sum)
        #输出这次试验累计的奖励
        reward_sum = 0 #奖励重新置为0
        env.reset()#重启环境


print ("随机测试结束")

# 超参数
H = 50  # 隐含的节点数
batch_size = 25  #
learning_rate = 1e-1  # 学习率
gamma = 0.99  # Reward的discount比例设为0.99,该值必须小于1.
#防止Reward被无损耗地不断累加导致发散，这样也能区分当前Reward和未来的Reward的价值
#当前Action直接带来的Reward不需要discount，而未来的Reward因存在不确定性，所以需要discount

D = 4  # 环境信息observation的维度D为4

tf.reset_default_graph()

#策略网络的具体结构。
#该网络将接受observation作为信息输入，最后输出一个概率值用以选择Action
#这里只有两个Action，向左施加力或者向右施加力，因此可以通过一个概率值决定

observations = tf.placeholder(tf.float32, [None, D], name="input_x")
#创建输入信息observations的placeholder其维度为D

#使用tf.contrib.layers.xavier_initializer方法初始化隐含层的权重W1，其维度为[D,H]
W1 = tf.get_variable("W1", shape=[D, H],
                     initializer=tf.contrib.layers.xavier_initializer())

layer1 = tf.nn.relu(tf.matmul(observations, W1))
#接着使用tf.matmul将环境信息observations乘上W1再使用relu激活函数处理得到隐含层的输出layer1

#使用tf.contrib.layers.xavier_initializer方法初始化隐含层的权重W2，其维度为[H,1]
W2 = tf.get_variable("W2", shape=[H, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)
#将隐含层输出layer1乘以W2后，使用Sigmoid激活函数处理得到最后的输出概率


# From here we define the parts of the network needed for learning a good policy.
tvars = tf.trainable_variables()#获取策略网络中全部可训练的参数tvars
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")
#定义人工设置的虚拟label的占位符input_y
#以及每个Action的潜在价值的占位符

#当Action取值为1的概率为probability(即策略网络输出的概率)
#当Action取值为0的概率为1-probability
#label取值与Action相反，即label=1-Action。
#当Action为1时，label为0，此时loglik=tf.log(probability),Action取值为1的概率的对数
#当Action为0时，label为1，此时loglik=tf.log(1-probability),Action取值为0的概率的对数
#所以，loglik其实就是当前Action对应的概率的对数
loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))

loss = -tf.reduce_mean(loglik * advantages)
#将loglik与潜在价值advanages相乘，并取负数作为损失，即优化目标
newGrads = tf.gradients(loss, tvars)
#使用tf.gradients求解模型参数关于loss的梯度

# Once we have collected a series of gradients from multiple episodes, we apply them.
# We don't just apply gradeients after every episode in order to account for noise in the reward signal.
#模型的优化器使用Adam算法
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Our optimizer
W1Grad = tf.placeholder(tf.float32,
                        name="batch_grad1")  # Placeholders to send the final gradients through when we update.
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
#我们分别设置两层神经网络参数的梯度的placeholder

batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))
#并使用adam.apply_gradients定义我们更新模型参数的操作updateGrads
#之后计算参数的梯度，当积累到一定样本量的梯度，就传入W1Grad和W2Grad，并执行updateGrads更新模型参数
#注意：
#深度强化学习的训练和其他神经网络一样，也使用batch training的方式。
#我们不逐个样本的更新参数，而是累计到一个batch_size的样本的梯度在更新参数
#防止单一样本随机扰动的噪声对模型带来不良影响。


#用来估算每一个Action对应的潜在价值discount_r
#因为CartPole问题中每次获得的Reward都和前面的Action有关，属于delayed reward
#因此需要比较精确地衡量每一个Action实际带来的价值，不能只看当前这一步的Reward，而要考虑后面的Delayed Reward
#哪些能让Pole长时间保持在空中竖直的Action应该拥有较大的值，而哪些最终导致pole倾倒的Action，应该拥有较小的期望价值。
#我们判断越靠后的Aciton的期望价值越小，因为它们更可能是导致Pole倾倒的原因，并且判断越考前的期望价值约大，因为它们
#长时间保持了Pole的竖直，和整倾倒的原因没那么大
#在CartPole问题中，除了最后结束时刻的Action为0,其余的均为1，。
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    #定义每个Action除直接获得的Reward外的潜在价值running_add
    #running_add是从后向前累计的，并且需要经过discount衰减。
    #每一个Action的潜在价值，即为后一个Action的前在价值乘以衰减系数gamma，再加上它直接获得的reward
    #即running_add * gamma + r[t]
    #这样从最后一个Action不断向前累计计算，即可得到全部Action的潜在价值。
    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

#xs为环境信息observation的列表
#ys为我们定义的label的列表
#drs为我们记录的每一个Action的Reward
xs, ys, drs = [], [], []
# running_reward = None
reward_sum = 0  #累计的Reward
episode_number = 1
total_episodes = 10000 #总试验次数
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:#创建sess
    rendering = False #render的标志关闭，因为会带来较大的延迟
    sess.run(init)#初始化全部参数

    observation = env.reset()  # Obtain an initial observation of the environment
    #先初始化CartPole的环境并获得初始状态

    # Reset the gradient placeholder. We will collect gradients in
    # gradBuffer until we are ready to update our policy network.
    gradBuffer = sess.run(tvars)
    #获取所有模型参数,用来创建储存参数梯度的缓冲器gradBuffer

    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
        #将gradBuffer全部初始化为0

    #接下来，每次试验中，我们将收集参数的梯度存储到gradBuffer中，直到完成了一个batch_size的试验
    #再将汇总的梯度更新到模型参数
    while episode_number <= total_episodes:

        # Rendering the environment slows things down,
        # so let's only look at it once our agent is doing a good job.
        #当某个batch的平均Reward达到100以上时，即Agent表现良好。
        if reward_sum / batch_size > 100 or rendering == True:
            #调用env.render()对试验环境进行展示
            env.render()
            rendering = True

        # Make sure the observation is in a shape the network can handle.
        #先使用tf.reshape将observation变形为策略网络的输入的格式
        x = np.reshape(observation, [1, D])

        # Run the policy network and get an action to take.
        #然后传入网络中
        tfprob = sess.run(probability, feed_dict={observations: x})
        #获得网络输出的概率tfprob,即Action取值为1的概率

        action = 1 if np.random.uniform() < tfprob else 0
        #接下来，我们在0到1之间随机抽样，若随机值小于tfprob，则令Action取值为1,
        #否则令Action取值为0,

        xs.append(x)  # 将输入的环境信息observation添加到列表xs中
        y = 1 if action == 0 else 0  # a "fake label" 虚拟的label
        ys.append(y)#添加到列表ys中

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        #使用env.step执行一次Action，获取observation,reward,done和info

        reward_sum += reward
        #将reward累加到reward_sum

        drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)
        #并将reward添加到列表drs中

        if done:#为True时，表示一次试验结束
            episode_number += 1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            #使用np.vstack将几个列表xs,ys,drs中的元素纵向堆叠起来

            xs, ys, drs = [], [], []  # 清空，以备下次使用


            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            # 使用前面定义好的discount_rewards函数计算每一步Action的潜在价值
            #并进行标准化，得到一个零均值，标准差为1的分布
            #discount_reward会参与到模型损失的计算，分布稳定的discount_reward有利于训练的稳定

            # Get the gradient for this episode, and save it in the gradBuffer
            #将epx,epy,discounted_epr输入网络，并求解梯度
            tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad
                #将获得的梯度累加到gradBuffer中去

            # If we have completed enough episodes, then update the policy network with our gradients.
            if episode_number % batch_size == 0:
                #当进行试验的次数达到batch_size的整数倍时，gradBuffer中就积累了足够多的梯度。
                #因此使用updateGrads操作将gradBuffer中的梯度更新到策略网络的模型参数中去
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
                    #清空gradBuffer为计算下一个batch的梯度做准备

                # Give a summary of how well our network is doing for each batch of episodes.
                # running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print('Average reward for episode %d : %f.' % (episode_number, reward_sum / batch_size))

                if reward_sum / batch_size > 200:
                    print("Task solved in", episode_number, 'episodes!')
                    break

                reward_sum = 0

            observation = env.reset()
            #每次试验结束后，将任务环境env重置，方便下一次试验