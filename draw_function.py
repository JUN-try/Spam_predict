from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_decision_regions(X,y,classifier=False,x_label='X',y_label='Y',title=' ',resolution=0.02,size=(8,8)):
	
    
    #サイズ
    fig,axes=plt.subplots(figsize=size)


    #マーカーとカラーマップの準備
    markers=('s','x','o','^','V')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])


    
    
    
    if(classifier):
        #決定領域のプロット
        x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
        y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
        #グリッドポイントの生成
        xx,yy=np.meshgrid(np.arange(x_min,x_max,resolution),np.arange(y_min,y_max,resolution))
        #各特徴量を一次元配列に変換して予測を実行
        Z=classifier.predict(np.array([xx.ravel(),yy.ravel()]).T)
        #予測結果を元のグリッドポイントのデータサイズに変更
        Z=Z.reshape(xx.shape)
        #グリッドポイントの等高線のプロット
        plt.contourf(xx,yy,Z,alpha=0.4,cmap=cmap)
        #軸の範囲の設定
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
    
    
    #クラスごとにサンプルをプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl,0],y=X[y == cl,1],alpha=0.6,c=cmap(idx),edgecolor='black',marker=markers[idx],label=cl)
        											
        																		
	#軸ラベルの設定
    plt.xlabel(x_label)
    plt.ylabel(y_label)


    plt.legend(loc='best')

    plt.title(title)
    plt.show()



    
def plot_scatter(x,y,title=None,x_label='X',y_label='Y',size=(8,8)):
    


    fig,axes=plt.subplots(figsize=size)
    
    plt.scatter(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.show()



