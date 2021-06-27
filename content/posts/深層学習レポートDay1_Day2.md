---
title: "深層学習前半レポート"
date: 2021-06-27T16:15:35+09:00
draft: false
mathjax: true
---




# 1. 深層学習day1レポート
## 1.0 ニューラルネットワークの全体像
##### 確認テスト1
ディープラーニングは、結局何をやろうとしているか2行以内で述べよ。また、次の中のどの値の最適化が最終目的か。全て選べ。（1分）
①入力値[ X] ②出力値[ Y]③重み[W]④バイアス[b]⑤総入力[u] ⑥中間層入力[ z]⑦学習率[ρ]

##### 回答
多数の中間層を用いることにより、入力から目的とする数値を出力する変換を行う数学モデルを構築すること。③と④。

##### 確認テスト2
次のネットワークを紙にかけ。  
入力層 2ノード1層  
中間層 ３ノード2層  
出力層 1ノード1層（5分）  
NN全体像確認テスト

##### 回答
{{< figure src="/image/確認テスト2.png" title="ネットワーク" class="center" width="600" height="300" >}}

## 1.1 入力層〜中間層

ある入力の特徴の入力を各ノードに代入し、その入力に対して重みwとバイアスbを考慮し、中間層のノードに対して出力を行う。動物の写真判定の例では、入力として「動物の体長」や「動物の足の長さ」など、写真を判定するのに必要な情報をインプットする。

##### 確認テスト3
この図式に動物分類の実例を入れてみよう。（3分）
{{< figure src="/image/確認テスト3問題.png" title="確認テスト問題" class="center" width="600" height="300" >}}

##### 回答
{{< figure src="/image/確認テスト3回答.png" title="確認テスト回答" class="center" width="300" height="600" >}}

##### 確認テスト4
この数式をPythonで書け。（2分）

##### 回答
{{< figure src="/image/確認テスト3問題.png" title="確認テスト問題" class="center" width="600" height="300" >}}
```python
u = np.dot(x,W) + b
```

##### 確認テスト5
1-1のファイルから中間層の出力を定義しているソースを抜き出せ。（2分）

##### 回答
``` python
# 中間層出力
z = functions_dnn.relu(u)
print_vec("中間層出力", z)
```

### 実装演習
単層単ユニットにおける実装演習結果を示す。
```python 
#functionsモジュールを呼び出すことができなかったため、functionsをfunctions_dnnに読み替えて実装

import numpy as np
#from common import functions_dnn
import functions_dnn


def print_vec(text, vec):
    print("*** " + text + " ***")
    print(vec)
    #print("shape: " + str(x.shape))
    print("")


# 順伝播（単層・単ユニット）

# 重み
W = np.array([[0.1], [0.2]])

## 試してみよう_配列の初期化
#W = np.zeros(2)
#W = np.ones(2)
#W = np.random.rand(2)
#W = np.random.randint(5, size=(2))

print_vec("重み", W)


# バイアス
b = 0.5

## 試してみよう_数値の初期化
#b = np.random.rand() # 0~1のランダム数値
#b = np.random.rand() * 10 -5  # -5~5のランダム数値

print_vec("バイアス", b)

# 入力値
x = np.array([2, 3])
print_vec("入力", x)


# 総入力
u = np.dot(x, W) + b
print_vec("総入力", u)

# 中間層出力
z = functions_dnn.relu(u)
print_vec("中間層出力", z)

```
[出力結果]  
*** 重み ***
[[0.1]
 [0.2]]  
*** バイアス ***
0.5  
*** 入力 ***
[2 3]  
*** 総入力 ***
[1.3]  
*** 中間層出力 ***
[1.3]


## 1.2 活性化関数

ニューラルネットワークにおいて、次のそうへの出力の大きさを決める"非線形な関数"のことを、活性化関数という。
入力値の値によっては、次の層への信号のON/OFFや強弱を定める働きを持っている。

* 中間層用の活性化関数
  * ReLU関数  
今最も使われている活性化関数でで、勾配消失問題の回避とスパース化に貢献することでいい成果をもたらしている。

  * シグモイド（ロジスティック）関数  
0~1を緩やかに変化する関数で、ステップ関数ではON/OFFしかない状態に対し、信号の強弱を伝えられるようになり、予想ニューラルネットワーク普及のきっかけとなった。

  * ステップ関数  
閾値を超えたら発火する関数でさり、出力は常に1か0である。パーセプトロン（NNの前進）で利用された関数。

* 出力層用の活性化関数
  * ソフトマックス関数
  * 恒等写像
  * シグモイド関数（ロジスティック）関数



##### 確認テスト1
線形と非線形の違いを図に書いて、簡易に説明せよ。

##### 回答
線形：いわゆる比例の関係。二次元上では直線で表すことができる。  
非線形：二次元上で直線で表すことができない関数。


##### 確認テスト2
配布されたソースコードより、該当する箇所を抜き出せ。

##### 回答
{{< figure src="/image/確認テスト1.2.2問題.png" title="確認テスト問題" class="center" width="600" height="300" >}}
```python
z = functions_dnn.sigmoid(u)
```

### 実装演習
common関数内に存在する活性化関数について、実装演習を行う。

```python
import numpy as np

# 中間層の活性化関数
# シグモイド関数（ロジスティック関数）
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# ReLU関数
def relu(x):
    return np.maximum(0, x)

# ステップ関数（閾値0）
def step_function(x):
    return np.where( x > 0, 1, 0) 

```

## 1.3 出力層
### 誤差関数

出力層では、ニューラルネットワークによって得られた出力結果（出力値）と訓練データ（正解値）との差分を誤差関数として定義している。

\begin{eqnarray*}
 E_n(\boldsymbol{w}) = \frac{1}{2} \sum^J_{j=1} (y_j - d_j)^2 = \frac{1}{2} ||(\boldsymbol{y} - \boldsymbol{d})||^2 
\end{eqnarray*}
 
### 全結合NN - 出力層の活性化関数
出力層と中間層の活性化関数の違いとして以下の特徴があるため、出力層と中間層で利用される活性化関数は異なっている。

* 値の強弱
  * 中間層：閾値の前後で信号の強弱を調整
  * 出力層：信号の大きさ（比率）はそのままに変換

* 確率出力
  * 分類問題の場合、出力層の出力は0~1の範囲に限定し、総和を1とする必要がある。

##### 確認テスト1
なぜ引き算ではなく二乗するかを述べよ、また、下式の1/2はどういう意味を持つか述べよ。
\begin{eqnarray*}
 E_n(\boldsymbol{w}) = \frac{1}{2} \sum^J_{j=1} (y_j - d_j)^2 = \frac{1}{2} ||(\boldsymbol{y} - \boldsymbol{d})||^2 
\end{eqnarray*}

##### 回答
二乗しない場合、正負の値が混在するものを総和するため、誤差としての特徴を抽出するのが難しいため。  
1/2の意味は、逆伝播法にて誤差関数を微分して重みをアップデートする際に、計算が容易になるため。

##### 確認テスト2
{{< figure src="/image/確認テスト1.3.2問題.png" title="確認テスト問題" class="center" width="600" height="300" >}}

##### 回答
```python
# ソフトマックス関数
def softmax(x):
  if x.ndim == 2:  # ミニバッチとしてデータを取扱う際に利用する
    x = x.T        
    x = x - np.max(x, axis=0) # オーバーフロー対策
    y = np.exp(x) / np.sum(np.exp(x), axis=0) # ①を求めている、分母が③で分子が②
    return y.T

  x = x - np.max(x)  # オーバーフロー対策
  return np.exp(x) / np.sum(np.exp(x)) # ①を求めている、分母が③で分子が②
```

##### 確認テスト3
{{< figure src="/image/確認テスト1.3.3問題.png" title="確認テスト問題" class="center" width="600" height="300" >}}

##### 回答
```python
# クロスエントロピー
def cross_entropy_error(d, y):
  if y.ndim == 1:    # 1次元の場合。
    d = d.reshape(1, d.size)    # (1, 要素数)のベクトルへ変形する
    y = y.reshape(1, y.size)    # (1, 要素数)のベクトルへ変形する

  # 教師データが one-hot-vector の場合、正解ラベルのインデックスに変換
  if d.size == y.size:
    d = d.argmax(axis=1)  # 最大値のインデックス値を取得

  batch_size = y.shape[0] # バッチサイズを定義
  return -np.sum(np.log(y[np.arange(batch_size), d] + 1e-7) / batch_size 
  # バッチサイズ分、対数関数に与えている。
```

### 実装演習
出力層での活性化関数についての実装結果を以下に記載。
```python
# 出力層の活性化関数
# ソフトマックス関数
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

# ソフトマックスとクロスエントロピーの複合関数
def softmax_with_loss(d, x):
    y = softmax(x)
    return cross_entropy_error(d, y)

# 誤差関数
# 平均二乗誤差
def mean_squared_error(d, y):
    return np.mean(np.square(d - y)) / 2

# クロスエントロピー
def cross_entropy_error(d, y):
    if y.ndim == 1:
        d = d.reshape(1, d.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if d.size == y.size:
        d = d.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), d] + 1e-7)) / batch_size
```

## 1.4 勾配降下法

深層学習の目的としては、学習を通して誤差関数を最小とするネットワークを作成することである。これはつまり、誤差関数$E(\boldsymbol{w})$を最小化するパラメータ$\boldsymbol{w}$を見つけることである。その際に、勾配降下法を用いて、パラメータ$\boldsymbol{w}$を求める。

* 勾配降下法
* 確率的勾配降下法
* ミニバッチ勾配降下法

勾配降下法で利用されているアルゴリズムとして、以下のものがある。
* Momentoum
* AdaGrad
* AdaDelta
* Adam

勾配降下法では、学習率$\epsilon$によって学習の効率が大きく異なる。学習率が大きすぎた場合、最小値にいつまでもたどり着かず発散してしまうが、学習率が小さい場合発散することはないが、小さすぎると収束するまでに時間がかかってしまう。そのため、発散せずかつ時間をそれほど要さないような適切な学習率を設定することが、勾配降下法では必要とされる。

##### 確認テスト1
{{< figure src="/image/確認テスト1.4.1問題.png" title="確認テスト問題" class="center" width="600" height="150" >}}

##### 回答
```python
network[key]  -= learning_rate* grad[key]
grad = backward(x, d, z1, y)S4)
```

##### 確認テスト2
オンライン学習とは何か2行でまとめよ（2分）

##### 回答
ニューラルネットワークの入力に対して、その都度データを与えて学習をさせる方法。学習データが入ってくる都度パラメータを更新していき、学習を進めていく。一方、バッチ学習は、一度に全ての学習データを投入する手法。

### 実装演習
誤差逆伝播法における実装演習と合わせて実施。


## 1.5 誤差逆伝播法
誤差逆伝播法では、算出された誤差を出力層側から順に微分し、前の層前の層へと伝播していく。最小限の計算で各パラメータでの微分値を解析的に計算する手法である。計算結果（=誤差）から微分を逆算することで、不要な再帰的計算を避けて微分を算出することができる。

##### 確認テスト1
誤差逆伝播法では不要な再帰的処理を避ける事が出来る。既に行った計算結果を保持しているソースコードを抽出せよ。

##### 回答
```python
    # １回使った微分を使いまわしている。
    delta1 = np.dot(delta2, W2.T) * functions.d_sigmoid(z1)

    delta1 = delta1[np.newaxis, :]
```

##### 確認テスト2
2つの空欄に該当するソースコードを探せ
```python
delta2 = functions.d_mean_squared_error(d, y)
grad['W2'] = np.dot(z1.T, delta2)
```

### 実装演習
誤差逆伝播法を利用して重みを更新しているソースに関する演習を実施する。
```python
import numpy as np
import functions_dnn # functions.pyをリネームしている
import matplotlib.pyplot as plt

def print_vec(text, vec):
    print("*** " + text + " ***")
    print(vec)
    #print("shape: " + str(x.shape))
    print("")


# ウェイトとバイアスを設定
# ネートワークを作成
def init_network():
    print("##### ネットワークの初期化 #####")

    network = {}
    network['W1'] = np.array([
        [0.1, 0.3, 0.5],
        [0.2, 0.4, 0.6]
    ])

    network['W2'] = np.array([
        [0.1, 0.4],
        [0.2, 0.5],
        [0.3, 0.6]
    ])

    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['b2'] = np.array([0.1, 0.2])
    
    print_vec("重み1", network['W1'])
    print_vec("重み2", network['W2'])
    print_vec("バイアス1", network['b1'])
    print_vec("バイアス2", network['b2'])

    return network

# 順伝播
def forward(network, x):
    print("##### 順伝播開始 #####")

    W1, W2 = network['W1'], network['W2']
    b1, b2 = network['b1'], network['b2']
    
    u1 = np.dot(x, W1) + b1
    z1 = functions_dnn.relu(u1)
    u2 = np.dot(z1, W2) + b2
    y = functions_dnn.softmax(u2)
    
    print_vec("総入力1", u1)
    print_vec("中間層出力1", z1)
    print_vec("総入力2", u2)
    print_vec("出力1", y)
    print("出力合計: " + str(np.sum(y)))

    return y, z1

# 誤差逆伝播
def backward(x, d, z1, y):
    print("\n##### 誤差逆伝播開始 #####")

    grad = {}

    W1, W2 = network['W1'], network['W2']
    b1, b2 = network['b1'], network['b2']
    #  出力層でのデルタ
    delta2 = functions_dnn.d_sigmoid_with_loss(d, y)
    #  b2の勾配
    grad['b2'] = np.sum(delta2, axis=0)
    #  W2の勾配
    grad['W2'] = np.dot(z1.T, delta2)
    #  中間層でのデルタ
    delta1 = np.dot(delta2, W2.T) * functions_dnn.d_relu(z1)
    # b1の勾配
    grad['b1'] = np.sum(delta1, axis=0)
    #  W1の勾配
    grad['W1'] = np.dot(x.T, delta1)
        
    print_vec("偏微分_dE/du2", delta2)
    print_vec("偏微分_dE/du2", delta1)

    print_vec("偏微分_重み1", grad["W1"])
    print_vec("偏微分_重み2", grad["W2"])
    print_vec("偏微分_バイアス1", grad["b1"])
    print_vec("偏微分_バイアス2", grad["b2"])

    return grad
    
# 訓練データ
x = np.array([[1.0, 5.0]])
# 目標出力
d = np.array([[0, 1]])
#  学習率
learning_rate = 0.01
network =  init_network()
y, z1 = forward(network, x)

# 誤差
loss = functions_dnn.cross_entropy_error(d, y)

grad = backward(x, d, z1, y)
for key in ('W1', 'W2', 'b1', 'b2'):
    network[key]  -= learning_rate * grad[key]

print("##### 結果表示 #####")    


print("##### 更新後パラメータ #####") 
print_vec("重み1", network['W1'])
print_vec("重み2", network['W2'])
print_vec("バイアス1", network['b1'])
print_vec("バイアス2", network['b2'])
```
[出力結果]
##### ネットワークの初期化 #####
*** 重み1 ***
[[0.1 0.3 0.5]
 [0.2 0.4 0.6]]

*** 重み2 ***
[[0.1 0.4]
 [0.2 0.5]
 [0.3 0.6]]

*** バイアス1 ***
[0.1 0.2 0.3]

*** バイアス2 ***
[0.1 0.2]

##### 順伝播開始 #####
*** 総入力1 ***
[[1.2 2.5 3.8]]

*** 中間層出力1 ***
[[1.2 2.5 3.8]]

*** 総入力2 ***
[[1.86 4.21]]

*** 出力1 ***
[[0.08706577 0.91293423]]

出力合計: 1.0

##### 誤差逆伝播開始 #####
*** 偏微分_dE/du2 ***
[[ 0.08706577 -0.08706577]]

*** 偏微分_dE/du2 ***
[[-0.02611973 -0.02611973 -0.02611973]]

*** 偏微分_重み1 ***
[[-0.02611973 -0.02611973 -0.02611973]
 [-0.13059866 -0.13059866 -0.13059866]]

*** 偏微分_重み2 ***
[[ 0.10447893 -0.10447893]
 [ 0.21766443 -0.21766443]
 [ 0.33084994 -0.33084994]]

*** 偏微分_バイアス1 ***
[-0.02611973 -0.02611973 -0.02611973]

*** 偏微分_バイアス2 ***
[ 0.08706577 -0.08706577]

##### 結果表示 #####
##### 更新後パラメータ #####
*** 重み1 ***
[[0.1002612  0.3002612  0.5002612 ]
 [0.20130599 0.40130599 0.60130599]]

*** 重み2 ***
[[0.09895521 0.40104479]
 [0.19782336 0.50217664]
 [0.2966915  0.6033085 ]]

*** バイアス1 ***
[0.1002612 0.2002612 0.3002612]

*** バイアス2 ***
[0.09912934 0.20087066]












