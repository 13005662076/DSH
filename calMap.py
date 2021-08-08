import numpy as np

'''
Bv：查询输入的汉明码，m*bit
Bt：检索集的汉明码，n**bit
queryL：查询输入的标签，m*numclasses(labels_onehot)
totalL：检索集的标签，n*numclasses(labels_onehot)
'''

#计算图像之间的向量距离
def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    #print(np.dot(B1, B2.transpose()))
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

#计算距离map值
def CalcMap(Bv, Bt, queryL, totalL):
    
    num_query = queryL.shape[0]
    map = 0
    
    #遍历查找每一张图片
    for iter in range(num_query):
        #获得图像是否相似的矩阵
        gnd = (np.dot(queryL[iter, :], totalL.transpose()) > 0).astype(np.float32)
        #获得检测图像和目标图像库相似的图片数量
        tsum = int(np.sum(gnd))
        
        if tsum == 0:
            continue
        #计算汉明距离
        hamm = CalcHammingDist(Bv[iter, :], Bt)

        #对获得的距离矩阵的内容进行从小到大排序，并获得排序后的下标ind
        ind = np.argsort(hamm)
        #使用ind对相关矩阵gnd进行排序
        gnd = gnd[ind]
        
        count = np.linspace(1, tsum, tsum)
        
        #获得相似矩阵gnd中为1的下标(相似为1，否则为0)
        tindex = np.asarray(np.where(gnd ==1)) + 1.0

        #count / tindex表示第i张相似的精确率
        map_ = np.mean(count / (tindex))
        
        map = map + map_
    map = map / num_query
    

    return map

def CalcTopMap(qB, rB, queryL, retrievalL, topk):
    
    num_query = queryL.shape[0]
    topkmap = 0
    
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    
    return topkmap

if __name__=='__main__':
    qB = np.array([[1,-1,1,1],[-1,1,-1,-1],[1,-1,-1,-1]])
    rB = rB = np.array([
        [ 1,-1,-1,-1],
        [-1, 1, 1,-1],
        [ 1, 1, 1,-1],
        [-1,-1, 1, 1],
        [ 1, 1,-1,-1],
        [ 1, 1, 1,-1],
        [-1, 1,-1,-1]])
    queryL = np.array([
        [1,0,0],
        [1,1,0],
        [0,0,1],
    ], dtype=np.int64)
    retrievalL = np.array([
        [0,1,0],
        [1,1,0],
        [1,0,1],
        [0,0,1],
        [0,1,0],
        [0,0,1],
        [1,1,0],
    ], dtype=np.int64)

    topk = 5
    map = CalcMap(qB, rB, queryL, retrievalL)
    #topkmap = CalcTopMap(qB, rB, queryL, retrievalL, topk)
    print(map)
    #print(topkmap)




#具体参考https://blog.csdn.net/weixin_37724529/article/details/118108881
#http://t.zoukankan.com/youmuchen-p-13547393.html
