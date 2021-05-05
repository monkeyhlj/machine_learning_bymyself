# machine_learning_bymyself
**机器学习记录**
## 目录

#### 决策树中的分类树1
    引入了graphviz，画图报错：https://blog.csdn.net/weixin_42067800/article/details/100751057
    [*zip(feature_name,clf.feature_importances_)]  #连成元组
    用random_state这个参数来设立随机数种子（随机模式20）因为之前它是随机选取一组特征来进行不纯度的计算
    在这里不得不提的是，所有接口中要求输入X_train和X_test的部分，输入的特征矩阵必须至少是一个二维矩阵。
sklearn不接受任何一维矩阵作为特征矩阵被输入。如果你的数据的确只有一个特征，那必须用reshape(-1,1)来给
矩阵增维；如果你的数据只有一个特征和一个样本，使用reshape(1,-1)来给你的数据增维。

#### 决策树中的回归树2
    #用numpy来生成一个随机数种子，跟分类树中的random_state差不多
    rng = np.random.RandomState(1)
    #rng.rand(10)，生成0-1之间的10个数；rng.rand(2,3),生成2*3矩阵
    #为什么要生成80*1的二维矩阵，因为训练集和测试集在训练时不能传入一维数组
    #5*，，变成0-5之间的80个数
    #np.sort排序，axis=0，y轴列
    X = np.sort(5 * rng.rand(80,1),axis=0)
    #用numpy中的正弦函数,此时y也是（80,1）
    #ravel()降维，降维后(80,)
    y = np.sin(X).ravel()
    y[::5]  #取出16个数  80/5=16
    #加上噪声
    #y[行:列:步长]  y[::5]所有行所有列每5个取一个数，给这16个数随机的加上一个数
    #rng.rand(16)：0-1之间  0.5-rng.rand(16)：-0.5-0.5之间  3*：扩大一点噪声：-1.5-1.5之间
    y[::5] += 3 * (0.5 - rng.rand(16))

    #np.arange(开始点，结束点，步长) []后面是一种增维的用法（见上例子）
    X_test = np.arange(0.0,5.0,0.01)[:,np.newaxis]
    X_test


#### 决策树中的回归树2（泰坦尼克号幸存者预测）
    data = pd.read_csv("./datasets/train.csv")
    data.info()  #查看数据的详细信息，可以看到有缺失值
    #删选特征 inplace=True会覆盖原表 =False则不会覆盖，但是要生成一个新的对象
    #axis=1删除列
    data = data.drop(['Cabin',"Name","Ticket"],inplace=False,axis=1)  

    #补充：当axis=0时，要删除指定行可以用labels标签，例如labels=[2,3]删除2、3行
    
    #处理缺失值
    data["Age"] = data["Age"].fillna(data["Age"].mean()) #用均值去填补
    #Embarked只有两行缺失值，直接把这两行删掉
    #删掉有缺失值的行
    data = data.dropna(axis=0)
    
    labels = data["Embarked"].unique().tolist()
    #apply()在这一列里执行括号里面的操作
    #把Embarked里面的值转换成她对应得索引（变成数字）
    data["Embarked"] = data["Embarked"].apply(lambda x:labels.index(x))
    #把True和False变成0 1
    (data["Sex"] == "male").astype("int")
    #性别也可以用Embarked的方法来转换，这里用一种新的方法
    # data["Sex"] = (data["Sex"] == "male").astype("int")
    data.loc[:,"Sex"] = (data["Sex"] == "male").astype("int")
    
    data.iloc[:,3] #取出Sex这一列
    
    x = data.iloc[:,data.columns != "Survived"]
    
    #把索引恢复
    for i in [Xtrain,Xtest,Ytrain,Ytest]:
        i.index = range(i.shape[0])
    
    学习曲线
    网格搜索

#### 随机森林1（分类）
#### 随机森林2（回归）
    用随机森林回归填补缺失值
    plt.figure(figsize=(12,6)) #画出画布
    ax = plt.subplot(111) #plt.subplot添加子图 111 第一行第一列第一个表

    
#### 机器学习中调参的基本思想
#### 随机森林实例：在乳腺癌数据上的调参
#### 特征工程1-数据预处理
#### 特征工程2-特征选择 feature_selection
#### 降维算法1-PCA
#### 降维算法2案例-PCA对手写数字数据集的降维
#### 逻辑回归1
#### 逻辑回归2-案例：用逻辑回归制作评分卡
#### 聚类算法KMeans
#### 聚类算法案例：聚类算法用于降维，KMeans的矢量量化应用
#### 支持向量机SVM（上）
#### 支持向量机SVM（下）

#### 深度学习入门：sklearn中的神经网络（注：https://blog.csdn.net/hhhmonkey/article/details/113526935）


**附加学习：Python爬虫基础-莫烦教学**
