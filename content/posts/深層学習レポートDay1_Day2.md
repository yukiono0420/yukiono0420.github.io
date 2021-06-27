---
title: "深層学習レポートDay1Day2"
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




# 2. 深層学習day2レポート
# 2.1 勾配消失問題について
誤差逆伝播法が下位置に進んでいくにつれて、勾配が緩やかになっていく。これは、勾配降下法による重みパラメータの更新では、逆伝播の中に活性化関数の微分が複数回登場するためである。sigmoid関数の微分では、最大値が0.25であり大きな値では出力の変化量が微小であることから、sigmoid関数の微分が複数回行われた後は、勾配が微小になってしまうことがある。その結果、下位（入力側）パラメータはほとんど変化しないため、訓練が最適解に収束しなくなってしまう。

* 勾配消失の解決法
  * 活性化関数の選択：ReLU関数
  * 重みの初期値設定
  * バッチ正規化

### 2.1.1 ReLU関数
今最も使われている活性化関数。勾配消失問題の回避とスパース化に貢献することで良い結果を残している。
※スパース化：中間層の重みパラメータのうち、必要な重みのみが選択的に抽出されること

### 2.1.2 重みの初期値設定
#### Xavier
重みの要素を前のレイヤのノード数の平方根で割る初期値の設定方法。  
（重みはベクトルのため、１つの要素には個性があるため、個性の洗濯が起きる

#### He
重みの要素を前のレイヤのノード数のルートで割ったものに、$\sqrt{2}$を乗算して初期値を設定する。

### 2.1.3 バッチ正規化
バッチ正規化とは、ミニバッチ単位で、入力値のデータの偏りを抑制する手法である。バッチ正規化は、活性化関数に値を渡す前後に、バッチ正規化の処理を孕んだ層を加える。


##### 確認テスト1
重みの初期値に0を設定すると、どのような問題が発生するか。簡潔に説明せよ。

##### 回答
すべての重みの値が均一に更新されることになることから、各ノードに設定されている重みをもつ意味がなく、正しい学習を行うことができなくなる。

##### 確認テスト2
一般的に考えられるバッチ正規化の効果を2点あげよ。

##### 回答
* ニューラルネットワークの学習中に中間層の重みの更新が安定化する（学習が安定化し学習スピードが向上する）
* 過学習の抑制

### 実装演習
まず、多層構造ニューラルネットワーククラスを作成する。
```python
import numpy as np
import layers
from collections import OrderedDict
import functions_dnn
from mnist import load_mnist
import matplotlib.pyplot as plt


class MultiLayerNet:
    '''
    input_size: 入力層のノード数
    hidden_size_list: 隠れ層のノード数のリスト
    output_size: 出力層のノード数
    activation: 活性化関数
    weight_init_std: 重みの初期化方法
    '''
    def __init__(self, input_size, hidden_size_list, output_size, activation='relu', weight_init_std='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.params = {}

        # 重みの初期化
        self.__init_weight(weight_init_std)

        # レイヤの生成, sigmoidとreluのみ扱う
        activation_layer = {'sigmoid': layers.Sigmoid, 'relu': layers.Relu}
        self.layers = OrderedDict() # 追加した順番に格納
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = layers.Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = layers.Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

        self.last_layer = layers.SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, d):
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]

        return self.last_layer.forward(y, d) + weight_decay

    def accuracy(self, x, d):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if d.ndim != 1 : d = np.argmax(d, axis=1)

        accuracy = np.sum(y == d) / float(x.shape[0])
        return accuracy

    def gradient(self, x, d):
        # forward
        self.loss(x, d)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grad = {}
        for idx in range(1, self.hidden_layer_num+2):
            grad['W' + str(idx)] = self.layers['Affine' + str(idx)].dW
            grad['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grad
```
勾配消失問題の具体例として、活性化関数にsigmoid関数を使用してみる。
```python
# データの読み込み
(x_train, d_train), (x_test, d_test) = load_mnist(normalize=True, one_hot_label=True)

print("データ読み込み完了")

network = MultiLayerNet(input_size=784, hidden_size_list=[40, 20], output_size=10, activation='sigmoid', weight_init_std=0.01)

iters_num = 2000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
accuracies_train = []
accuracies_test = []

plot_interval=10

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    d_batch = d_train[batch_mask]

    # 勾配
    grad = network.gradient(x_batch, d_batch)
    
    for key in ('W1', 'W2', 'W3', 'b1', 'b2', 'b3'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, d_batch)
    train_loss_list.append(loss)
    
    if (i + 1) % plot_interval == 0:
        accr_test = network.accuracy(x_test, d_test)
        accuracies_test.append(accr_test)        
        accr_train = network.accuracy(x_batch, d_batch)
        accuracies_train.append(accr_train)

        print('Generation: ' + str(i+1) + '. 正答率(トレーニング) = ' + str(accr_train))
        print('                : ' + str(i+1) + '. 正答率(テスト) = ' + str(accr_test))
        

lists = range(0, iters_num, plot_interval)
plt.plot(lists, accuracies_train, label="training set")
plt.plot(lists, accuracies_test,  label="test set")
plt.legend(loc="lower right")
plt.title("accuracy")
plt.xlabel("count")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
# グラフの表示
plt.show()
```
下図のように、学習を進めても正答率が上がっておらず、勾配消失問題を引き起こしていることがわかる。
{{< figure src="/image/勾配消失例.png" title="勾配消失している場合の正答率" class="center" width="500" height="300" >}}


続いて、活性化関数をReLU関数にして学習を実施する。
```python
# データの読み込み
(x_train, d_train), (x_test, d_test) = load_mnist(normalize=True, one_hot_label=True)

print("データ読み込み完了")

network = MultiLayerNet(input_size=784, hidden_size_list=[40, 20], output_size=10, activation='relu', weight_init_std=0.01)

iters_num = 2000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
accuracies_train = []
accuracies_test = []

plot_interval=10

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    d_batch = d_train[batch_mask]

    # 勾配
    grad = network.gradient(x_batch, d_batch)
    
    for key in ('W1', 'W2', 'W3', 'b1', 'b2', 'b3'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, d_batch)
    train_loss_list.append(loss)
    
    if (i + 1) % plot_interval == 0:
        accr_test = network.accuracy(x_test, d_test)
        accuracies_test.append(accr_test)        
        accr_train = network.accuracy(x_batch, d_batch)
        accuracies_train.append(accr_train)

        print('Generation: ' + str(i+1) + '. 正答率(トレーニング) = ' + str(accr_train))
        print('                : ' + str(i+1) + '. 正答率(テスト) = ' + str(accr_test))
        
        
lists = range(0, iters_num, plot_interval)
plt.plot(lists, accuracies_train, label="training set")
plt.plot(lists, accuracies_test,  label="test set")
plt.legend(loc="lower right")
plt.title("accuracy")
plt.xlabel("count")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
# グラフの表示
plt.show()
```
下図のように、学習を進めるにつれて正答率が上がっており、勾配消失問題が解消していることがわかる。
{{< figure src="/image/勾配消失解消.png" title="勾配消失が解消している場合の正答率" class="center" width="500" height="300" >}}


続いて、初期設定方法にXavierを使用し、再び学習をする。
```python
# データの読み込み
(x_train, d_train), (x_test, d_test) = load_mnist(normalize=True, one_hot_label=True)

print("データ読み込み完了")

network = MultiLayerNet(input_size=784, hidden_size_list=[40, 20], output_size=10, activation='sigmoid', weight_init_std='Xavier')

iters_num = 2000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
accuracies_train = []
accuracies_test = []

plot_interval=10

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    d_batch = d_train[batch_mask]

    # 勾配
    grad = network.gradient(x_batch, d_batch)
    
    for key in ('W1', 'W2', 'W3', 'b1', 'b2', 'b3'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, d_batch)
    train_loss_list.append(loss)
    
    if (i + 1) % plot_interval == 0:
        accr_test = network.accuracy(x_test, d_test)
        accuracies_test.append(accr_test)        
        accr_train = network.accuracy(x_batch, d_batch)
        accuracies_train.append(accr_train)
        
        print('Generation: ' + str(i+1) + '. 正答率(トレーニング) = ' + str(accr_train))
        print('                : ' + str(i+1) + '. 正答率(テスト) = ' + str(accr_test))
        

lists = range(0, iters_num, plot_interval)
plt.plot(lists, accuracies_train, label="training set")
plt.plot(lists, accuracies_test,  label="test set")
plt.legend(loc="lower right")
plt.title("accuracy")
plt.xlabel("count")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
# グラフの表示
plt.show()
```

下図のように、Xavierを使うと、学習が緩やかに進んでいくことがわかる。
{{< figure src="/image/Xavier.png" title="Xavier" class="center" width="500" height="300" >}}


## 2.2 学習率最適化手法について
学習率を学習が進めていくにつれて変化させていくのが、学習率最適化手法である。
学習率最適化手法には以下の4つの手法がある。

* モメンタム
* AdaGrad
* RMSProp
* Adam

### 2.2.1 モメンタム
誤差をパラメータで微分したものと学習率の積を減算した後、現在の重みに前回の重みを減算した値と慣性の積を加算する。
モメンタムのメリットは以下の2点。
* 局所的最適解にはならず、大域的最適解となる。
* 谷間についてから最も低い位置に行くまでの時間が早い

### 2.2.2 AdaGrad
誤差をパラメータで微分したものと、再定義した学習率の積を減算する。
AdaGradのメリットは、勾配が緩やかな斜面に対して、最適解に近づけるという点があるが、一方学習率が徐々に小さくなるため、鞍点問題を引き起こすことがある。

### 2.2.3 RMSProp
誤差をパラメータで微分したものと、再定義した学習率の積を減算する。AdaGradの改良版である。
RMSPropのメリットは以下の2点。
* 局所的最適解にはならず、大域的最適解となる。
* ハイパーパラメータの調整が必要な場合が少ない。

### 2.2.4 Adam
Adamはモメンタムの「過去の勾配の指数関数的減衰平均」と、RMSPropの「過去の勾配の2乗の指数関数的減衰平均」をそれぞれ孕んだ最適化アルゴリズムである。AdamはモメンタムとRMSPropのそれぞれのメリットを孕んだアルゴリズムのため、広く使用されている。

##### 確認テスト
モメンタム・AdaGrad・RMSPropの特徴をそれぞれ簡潔に説明せよ。

##### 回答
上記の2.2.1〜2.2.3を参照。

### 実装演習
実装演習レポートには、Adamの実装結果を記載する。
まずは、SGD（確率的勾配降下法）にて学習をさせた結果を記載する。学習をしてもうまくいっていないことがわかる。
```python
import numpy as np
from collections import OrderedDict
import layers
from mnist import load_mnist
import matplotlib.pyplot as plt
from multi_layer_net import MultiLayerNet


# データの読み込み
(x_train, d_train), (x_test, d_test) = load_mnist(normalize=True, one_hot_label=True)

print("データ読み込み完了")

# batch_normalizationの設定 =======================
# use_batchnorm = True
use_batchnorm = False
# ====================================================


network = MultiLayerNet(input_size=784, hidden_size_list=[40, 20], output_size=10, activation='sigmoid', weight_init_std=0.01,
                       use_batchnorm=use_batchnorm)

iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

train_loss_list = []
accuracies_train = []
accuracies_test = []

plot_interval=10

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    d_batch = d_train[batch_mask]

    # 勾配
    grad = network.gradient(x_batch, d_batch)
    
    for key in ('W1', 'W2', 'W3', 'b1', 'b2', 'b3'):
        network.params[key] -= learning_rate * grad[key]
        
        loss = network.loss(x_batch, d_batch)
        train_loss_list.append(loss)
    
    
    if (i + 1) % plot_interval == 0:
        accr_test = network.accuracy(x_test, d_test)
        accuracies_test.append(accr_test)        
        accr_train = network.accuracy(x_batch, d_batch)
        accuracies_train.append(accr_train)
        
        print('Generation: ' + str(i+1) + '. 正答率(トレーニング) = ' + str(accr_train))
        print('                : ' + str(i+1) + '. 正答率(テスト) = ' + str(accr_test))

        
lists = range(0, iters_num, plot_interval)
plt.plot(lists, accuracies_train, label="training set")
plt.plot(lists, accuracies_test,  label="test set")
plt.legend(loc="lower right")
plt.title("accuracy")
plt.xlabel("count")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
# グラフの表示
plt.show()
```
{{< figure src="/image/SGD.png" title="SGDで学習を実施してもうまく学習がされない" class="center" width="500" height="300" >}}

ここからは、Adamにて学習を実施する。Adamにて実施すると学習が進むにつれて、正答率が上がっており、正しく学習が進んでいることがわかる。
```python
# データの読み込み
(x_train, d_train), (x_test, d_test) = load_mnist(normalize=True, one_hot_label=True)

print("データ読み込み完了")

# batch_normalizationの設定 =======================
# use_batchnorm = True
use_batchnorm = False
# ====================================================

network = MultiLayerNet(input_size=784, hidden_size_list=[40, 20], output_size=10, activation='sigmoid', weight_init_std=0.01,
                       use_batchnorm=use_batchnorm)

iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999

train_loss_list = []
accuracies_train = []
accuracies_test = []

plot_interval=10

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    d_batch = d_train[batch_mask]

    # 勾配
    grad = network.gradient(x_batch, d_batch)
    if i == 0:
        m = {}
        v = {}
    learning_rate_t  = learning_rate * np.sqrt(1.0 - beta2 ** (i + 1)) / (1.0 - beta1 ** (i + 1))    
    for key in ('W1', 'W2', 'W3', 'b1', 'b2', 'b3'):
        if i == 0:
            m[key] = np.zeros_like(network.params[key])
            v[key] = np.zeros_like(network.params[key])
            
        m[key] += (1 - beta1) * (grad[key] - m[key])
        v[key] += (1 - beta2) * (grad[key] ** 2 - v[key])            
        network.params[key] -= learning_rate_t * m[key] / (np.sqrt(v[key]) + 1e-7)                
        
        loss = network.loss(x_batch, d_batch)
        train_loss_list.append(loss)        
        
    if (i + 1) % plot_interval == 0:
        accr_test = network.accuracy(x_test, d_test)
        accuracies_test.append(accr_test)        
        accr_train = network.accuracy(x_batch, d_batch)
        accuracies_train.append(accr_train)
        
        print('Generation: ' + str(i+1) + '. 正答率(トレーニング) = ' + str(accr_train))
        print('                : ' + str(i+1) + '. 正答率(テスト) = ' + str(accr_test))
                

lists = range(0, iters_num, plot_interval)
plt.plot(lists, accuracies_train, label="training set")
plt.plot(lists, accuracies_test,  label="test set")
plt.legend(loc="lower right")
plt.title("accuracy")
plt.xlabel("count")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
# グラフの表示
plt.show()
```
{{< figure src="/image/Adam.png" title="Adamで学習を実施するとうまくいく" class="center" width="500" height="300" >}}


## 2.3 過学習について
過学習とは、学習データにおける誤差とテストデータにおける誤差で乖離が発生することである。これは、学習データに特化しすぎたパラメータが設定され、汎化性能がないモデルと言える。過学習が起きる原因は、パラメータ数が多かったり、パラメータ値が適切でないなど、ネットワークの自由度が多いことが原因で発生する。

### 2.3.1 正則化
過学習を抑える方法として、正則化がある。これはネットワークの自由度（層数、ノード数、パラメータの値など）を制約することである。正則化手法としては以下の2つがある。
* L1正則化、L2正則化
* ドロップアウト

##### 確認テスト
下図について、L1正則化を表しているグラフはどちらか答えよ。

##### 回答
右の図（Lasso回帰）

### 実装演習
過学習が以下の学習では発生している。

```python
import numpy as np
from collections import OrderedDict
from common import layers
from data.mnist import load_mnist
import matplotlib.pyplot as plt
from multi_layer_net import MultiLayerNet
from common import optimizer


(x_train, d_train), (x_test, d_test) = load_mnist(normalize=True)

print("データ読み込み完了")

# 過学習を再現するために、学習データを削減
x_train = x_train[:300]
d_train = d_train[:300]

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10)
optimizer = optimizer.SGD(learning_rate=0.01)

iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
accuracies_train = []
accuracies_test = []

plot_interval=10


for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    d_batch = d_train[batch_mask]

    grad = network.gradient(x_batch, d_batch)
    optimizer.update(network.params, grad)

    loss = network.loss(x_batch, d_batch)
    train_loss_list.append(loss)

    if (i+1) % plot_interval == 0:
        accr_train = network.accuracy(x_train, d_train)
        accr_test = network.accuracy(x_test, d_test)
        accuracies_train.append(accr_train)
        accuracies_test.append(accr_test)

        print('Generation: ' + str(i+1) + '. 正答率(トレーニング) = ' + str(accr_train))
        print('                : ' + str(i+1) + '. 正答率(テスト) = ' + str(accr_test))        

lists = range(0, iters_num, plot_interval)
plt.plot(lists, accuracies_train, label="training set")
plt.plot(lists, accuracies_test,  label="test set")
plt.legend(loc="lower right")
plt.title("accuracy")
plt.xlabel("count")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
# グラフの表示
plt.show()
```
{{< figure src="/image/overfit.png" title="過学習が起きている" class="center" width="500" height="300" >}}

続いて、L2正則化を実施し、学習を実施してみる。しかし、依然として過学習は解消されていないように見える。
```python
from common import optimizer

(x_train, d_train), (x_test, d_test) = load_mnist(normalize=True)

print("データ読み込み完了")

# 過学習を再現するために、学習データを削減
x_train = x_train[:300]
d_train = d_train[:300]


network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10)


iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate=0.01

train_loss_list = []
accuracies_train = []
accuracies_test = []

plot_interval=10
hidden_layer_num = network.hidden_layer_num

# 正則化強度設定 ======================================
weight_decay_lambda = 0.1
# =================================================

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    d_batch = d_train[batch_mask]

    grad = network.gradient(x_batch, d_batch)
    weight_decay = 0
    
    for idx in range(1, hidden_layer_num+1):
        grad['W' + str(idx)] = network.layers['Affine' + str(idx)].dW + weight_decay_lambda * network.params['W' + str(idx)]
        grad['b' + str(idx)] = network.layers['Affine' + str(idx)].db
        network.params['W' + str(idx)] -= learning_rate * grad['W' + str(idx)]
        network.params['b' + str(idx)] -= learning_rate * grad['b' + str(idx)]        
        weight_decay += 0.5 * weight_decay_lambda * np.sqrt(np.sum(network.params['W' + str(idx)] ** 2))

    loss = network.loss(x_batch, d_batch) + weight_decay
    train_loss_list.append(loss)        
        
    if (i+1) % plot_interval == 0:
        accr_train = network.accuracy(x_train, d_train)
        accr_test = network.accuracy(x_test, d_test)
        accuracies_train.append(accr_train)
        accuracies_test.append(accr_test)
        
        print('Generation: ' + str(i+1) + '. 正答率(トレーニング) = ' + str(accr_train))
        print('                : ' + str(i+1) + '. 正答率(テスト) = ' + str(accr_test))               


lists = range(0, iters_num, plot_interval)
plt.plot(lists, accuracies_train, label="training set")
plt.plot(lists, accuracies_test,  label="test set")
plt.legend(loc="lower right")
plt.title("accuracy")
plt.xlabel("count")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
# グラフの表示
plt.show()
```
{{< figure src="/image/L2.png" title="過学習は解消されず" class="center" width="500" height="300" >}}


## 2.4 畳み込みニューラルネットワークの概念
畳み込み層では、画像の場合、縦・横・チャンネルの3次元のデータをそのまま学習し、次に伝えることができる。つまり、3次元の空間情報も学習できるような層が畳み込み層である。

* パディング  
画像の周囲に固定値(例えば0)を埋め込むこと。
パディングなしで畳み込みをすると元の画像より小さくなるが、パディングをすることでサイズを維持することが出来る。また、パディングなしの場合、画像の端の方は他の部分と比べて畳み込みに使われる回数が少なくなり、特徴として抽出されにくいが、パディングをすることによってより端の方も特徴を抽出できるようになる。

* ストライド  
フィルタをずらす間隔を指定する方法。何も指定しない場合は、1つずつずらしていく。

* チャンネル  
空間的な奥行きのことを指す。例えば、カラー画像の場合はRGBの3チャンネルに分けて畳み込みを行う。

* プーリング層
  * 最大値プーリング
  * 平均値プーリング  
対象領域の中からある1つの値を取得する層。
畳み込みを行った後にプーリングを行うことでそれらしい特徴を持った値のみを抽出することが可能となる。


##### 確認テスト
サイズ6×6の入力画像を、サイズ2×2のフィルタで畳み込んだ時の出力画像のサイズを答えよ。なおストライドとパディングは1とする。

##### 回答
7x7

### 実装演習
ここでは実際に畳み込みを行っているソースコードのみ、レポートとして報告をする。
```python
from common import optimizer

# データの読み込み
(x_train, d_train), (x_test, d_test) = load_mnist(flatten=False)

print("データ読み込み完了")

# 処理に時間のかかる場合はデータを削減 
x_train, d_train = x_train[:5000], d_train[:5000]
x_test, d_test = x_test[:1000], d_test[:1000]


network = SimpleConvNet(input_dim=(1,28,28), conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

optimizer = optimizer.Adam()

iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
accuracies_train = []
accuracies_test = []

plot_interval=10



for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    d_batch = d_train[batch_mask]
    
    grad = network.gradient(x_batch, d_batch)
    optimizer.update(network.params, grad)

    loss = network.loss(x_batch, d_batch)
    train_loss_list.append(loss)

    if (i+1) % plot_interval == 0:
        accr_train = network.accuracy(x_train, d_train)
        accr_test = network.accuracy(x_test, d_test)
        accuracies_train.append(accr_train)
        accuracies_test.append(accr_test)
        
        print('Generation: ' + str(i+1) + '. 正答率(トレーニング) = ' + str(accr_train))
        print('                : ' + str(i+1) + '. 正答率(テスト) = ' + str(accr_test))               

lists = range(0, iters_num, plot_interval)
plt.plot(lists, accuracies_train, label="training set")
plt.plot(lists, accuracies_test,  label="test set")
plt.legend(loc="lower right")
plt.title("accuracy")
plt.xlabel("count")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
# グラフの表示
plt.show()
```
{{< figure src="/image/CNN.png" title="畳み込みを使用した学習" class="center" width="500" height="300" >}}

## 2.5 最新のCNN
この章では、過学習を防ぐ施策である、AlexNetという手法について、学習した。
AlexNetは、Hinton 教授らのチームによって発表された物体認識のためのモデル（アーキテクチャ）である。
2012年にILSVRC（ImageNet Large Scale Visual Recognition Challenge）で優勝したモデル。

過学習を防ぐ施策であり、サイズ4096の全結合層の出力にドロップアウトを使用している。

2012年前までの画像分類チャレンジコンテストにおいて、画像から特徴量を抽出し、その特徴量を用いて画像の分類を行なっていた。当時、画像から特徴量を抽出する際に、人が、物体の色・輝度・形など特徴量を設計していた。そのため、いかに有効な特徴量を設計できることが、画像分類の性能を左右していた。それが、2012年に、人が特徴量を設計しなくても、十分なデータさえ存在すれば、機械自身が特徴量を見つけ出すことが AlexNet によって示された。

AlexNet は、深さが 8 層の畳み込みニューラルネットワークである。100万枚を超えるイメージで学習させた事前学習済みのネットワークを、ImageNetデータベースから読み込むことができます。この事前学習済みのネットワークは、イメージを 1000 個のオブジェクトカテゴリ (キーボード、マウス、鉛筆、多くの動物など) に分類できる。結果として、このネットワークは広範囲のイメージに対する豊富な特徴表現を学習している。

参考文献：  
https://axa.biopapyrus.jp/deep-learning/cnn/image-classification/alexnet.html  
https://jp.mathworks.com/help/deeplearning/ref/alexnet.html;jsessionid=e166d58061fcf83da3bfb7de7740  
Krizhevsky A, Sutskever I, Hinton GE. ImageNet classification with deep convolutional neural networks. NeurIPS. 2012. DOI: 10.1145/3065386


### 実装演習
googleで公開されているAlexNetのpythonモジュールを以下に示す。
```python
def alexnet_v2_arg_scope(weight_decay=0.0005):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      biases_initializer=tf.constant_initializer(0.1),
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
      with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
        return arg_sc

def alexnet_v2(inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='alexnet_v2',
               global_pool=False):
  with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=[end_points_collection]):
      net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
                        scope='conv1')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
      net = slim.conv2d(net, 192, [5, 5], scope='conv2')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
      net = slim.conv2d(net, 384, [3, 3], scope='conv3')
      net = slim.conv2d(net, 384, [3, 3], scope='conv4')
      net = slim.conv2d(net, 256, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

      # Use conv2d instead of fully_connected layers.
      with slim.arg_scope([slim.conv2d],
                          weights_initializer=trunc_normal(0.005),
                          biases_initializer=tf.constant_initializer(0.1)):
        net = slim.conv2d(net, 4096, [5, 5], padding='VALID',
                          scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(
            end_points_collection)
        if global_pool:
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
          end_points['global_pool'] = net
        if num_classes:
          net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                             scope='dropout7')
          net = slim.conv2d(net, num_classes, [1, 1],
                            activation_fn=None,
                            normalizer_fn=None,
                            biases_initializer=tf.zeros_initializer(),
                            scope='fc8')
          if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
          end_points[sc.name + '/fc8'] = net
      return net, end_points
alexnet_v2.default_image_size = 224
```

