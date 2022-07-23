import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
import re
from os import listdir

def get_img(path_name,set_loss):
    filepath = ['checkpoints/{}/loss_log.txt'.format(path_name), \
    'checkpoints/{}/testacc_log.txt'.format(path_name)]
    acc = []
    loss = []
    num = 0
    for k in range(2):
        with open(filepath[k]) as f:
            for i in f.readlines():
                if k: 
                    tmp = re.search(r'(?<=TEST ACC: \[).*?\%\]',i)
                    # print(tmp)
                    if tmp and tmp.group():
                        acc.append(float(tmp.group().split(' ')[0]))
                else: 
                    if i.split(' ')[-2] != '2022)':
                        num+=1
                        if num == set_loss:
                            num=0
                            loss.append(float(i.split(' ')[-2]))
    x = [i for i in range(len(acc))]
    loss = loss[:len(acc)]
    # print(loss)
    # plt.figure()
    # plt.plot(x,acc,label='accuracy')
    # plt.plot(x,loss,label='loss')
    # plt.savefig('acc-loss.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,acc,'red',label='accuracy')
    ax2 = ax.twinx()
    ax2.plot(x,loss,'blue',label='loss')
    ax.grid()
    ax.set_xlabel("epoch")
    ax.set_ylabel(" acc ")
    ax2.set_ylabel("loss")

    acc_max = np.argmax(acc)
    loss_min = np.argmin(loss)
    ax.annotate(str(acc_max)+","+str(acc[acc_max]),xy=(acc_max,acc[acc_max]),xytext=(acc_max,acc[acc_max]))
    ax2.annotate(str(loss_min)+","+str(loss[loss_min]),xy=(loss_min,loss[loss_min]),xytext=(loss_min,loss[loss_min]))
    plt.savefig('img/acc-loss-{}.png'.format(path_name))


if __name__=='__main__':
    # li = [f for f in listdir('checkpoints/')]
    # for i in li:
    #     get_img(i)
    get_img('coseg_vases_2',4)