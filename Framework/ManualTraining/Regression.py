import random

class Regression():
    def training_sgd_fit(self, x, y, alpha,theta_array):
        diff = [0, 0]
        error1 = 0
        m = len(x)
        #init the parameters to zero
        theta0 = theta_array[0]
        theta1 = theta_array[1]
        theta2 = theta_array[2]
        epoch = 0
        error_array = []
        epoch_array = []
        #calculate the parameters
        # 线性回归：h(x) = theta0  + theta1 * x[i][0] + theta2 * x[i][1]
        # 损失函数：累和 (1/2) *  (y - h(x)) ^ 2
        # theta0 = theta0 - (   -alpha * (y - h(x))* 1 )
        # theta1 = theta1 - (   -alpha * (y - h(x))* x[i][0] )
        # theta2 = theta2 - (   -alpha * (y - h(x))* x[i][1] )
        # 1. 随机梯度下降算法在迭代的时候，每迭代一个新的样本，就会更新一次所有的theta参数。
        i = random.randint(0, m - 1)

        # (y - h(x))
        diff[0] = y[i] - (theta0 * 1 + theta1 * x[i][0] + theta2 * x[i][1])
        # - (y - h(x))x
        gradient0 = -diff[0] * 1
        gradient1 = -diff[0] * x[i][0]
        gradient2 = -diff[0] * x[i][1]
        # theta = theta - (  - alpha * (y - h(x))x )
        theta0 = theta0 - alpha * gradient0
        theta1 = theta1 - alpha * gradient1
        theta2 = theta2 - alpha * gradient2
        #theta3
        #calculate the cost function
        error1 = 0
        # 此处error为一个相对的Error值。
        for i in range(m):
            error1 += (y[i] - (theta0 * 1 + theta1 * x[i][0] + theta2 * x[i][1]))**2

        error1 = error1 / m
        return error1,(theta0,theta1,theta2)

    def training_bgd_fit(self, x, y, alpha,theta_array):
        diff = [0,0]
        error1 = 0
        error0 =0
        m = len(x)

        #init the parameters to zero
        theta0 = theta_array[0]
        theta1 = theta_array[1]
        theta2 = theta_array[2]
        sum0 = 0
        sum1 = 0
        sum2 = 0

        epoch = 0
        error_array = []
        epoch_array = []
        #calculate the parameters
        # 线性回归：hi(x) = theta0 + theta1 * x[i][1] + theta2 * x[i][2]  
        # 损失函数：(1/2) 累加 * (y - h(x)) ^ 2
        # theta = theta - 累和(  - alpha * (y - h(x))x )
        # 1. 随机梯度下降算法在迭代的时候，每迭代一个新的样本，就会更新一次所有的theta参数。
        #calculate the parameters
        # 2. 批梯度下降算法在迭代的时候，是完成所有样本的迭代后才会去更新一次theta参数
        for i in range(m):
            #begin batch gradient descent
            diff[0] = y[i]-( theta0 + theta1 * x[i][0] + theta2 * x[i][1] )
            sum0 = sum0 - ( -alpha * diff[0]* 1)
            sum1 = sum1 - ( -alpha * diff[0]* x[i][0])
            sum2 = sum2 - ( -alpha * diff[0]* x[i][1])
            #end  batch gradient descent
            
        theta0 = theta0 + sum0 / m;
        theta1 = theta1 + sum1 / m;
        theta2 = theta2 + sum2 / m;

        sum0 = 0
        sum1 = 0
        sum2 = 0
        #calculate the cost function
        error1 = 0
        for i in range(m):
            error1 += ( y[i]-( theta0 + theta1 * x[i][0] + theta2 * x[i][1] ) )**2               
        error1 = error1 / m
        return error1,(theta0,theta1,theta2)