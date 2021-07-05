---
title: "深層学習レポートDay3"
date: 2021-07-04T23:36:41+09:00
draft: false
mathjax: true
---

# 0. 深層学習全体像の復習

## 最新のCNN: AlexNet

AlexNetとは2012年威開かれた画像認識コンペティションで2位に大差をつけて優勝したモデルである。AlexNetの登場で、ディープラーニングが大きな注目を集めたきっかけとなった。AlexNetのモデル構造は、5層の畳み込み層およびプーリング層など、それに続く3層の全結合層から構成される。AlexNetでは、過学習を防ぐ施策として、サイズ4096の全結合層の出力にドロップアウトを使用している。

##### 確認テスト
サイズ5×5の入力画像を、サイズ3×3のフィルタで畳み込んだ時の出力画像のサイズを答えよ。なおストライドは2、パディングは1とする。

##### 回答
3×3


# Section1. 再帰型ニューラルネットワークについて

## 1.1 RNN全体像

### 1.1.1 RNNとは

RNN（再帰型ニューラルネットワーク）とは、時系列データに対応可能な、ニューラルネットワークである。

### 1.1.2 時系列データ

時系列データとは、時間的順序をおって一定間隔ごとに観測され、さらに相互に統計的依存関係が認められるようなデータ系列のことを指す。
* 具体例  
  * 音声データ  
  * テキストデータ

### 1.1.3 RNNの全体像
RNNでは、時刻t+1における中間層の重み$z_{t+1}$更新の際に、時刻tにおける中間層の重み$z_t$を入力データとして使用している。RNNの特徴としては、時系列モデルを扱うには、初期の状態と過去の時間t-1の状態を保持し、そこから次の時間でのtを再帰的に求める再帰構造が必要になる。

{{< figure src="/image/RNN構造.png" title="RNN構造" class="center" width="670" height="300" >}}
{{< figure src="/image/RNN数式.png" title="RNNを数式で表したもの" class="center" width="600" height="300" >}}

##### 確認テスト1
RNNのネットワークには大きくわけて3つの重みがある。1つは入力から現在の中間層を定義する際にかけられる重み、1つは中間層から出力を定義する際にかけられる重みである。残り1つの重みについて説明せよ。

##### 回答
1つ前の時点の中間層から、現在の中間層を定義する際にかけられる重み



## 1.2 BPTT(Backpropagation Through Time)
### 1.2.1 BPTTとは
BPTT(Backpropagation Through Time)とは、RNNにおいてのパラメータの調整方法の一つであり、時間軸方向への誤差逆伝播法である。

##### 確認テスト2
連鎖律の原理を使い、dz/dxを求めよ。

##### 回答
dz/dx = dz/dt * dt/dx = 2t * 1 = 2(x + y)

### 1.2.2 BPTTの数学的記述
パラメータの更新式は以下のようになる。
{{< figure src="/image/BPTT数式.png" title="BPTTにおける各パラメータの更新式" class="center" width="400" height="300" >}}

##### 確認テスト3
下図のy1をx・s0・s1・win・w・woutを用いて数式で表せ。※バイアスは任意の文字で定義せよ。※また中間層の出力にシグモイド関数g(x)を作用させよ。

##### 回答
\begin{eqnarray*}
y_1 &=& g(v_1) = g(W_{out}z_1 + c ) \\
&=& g(W_{out} f(W_{in} x_1 + W z_0 + b ) + c)
\end{eqnarray*}

### Section1 実装演習
2進数の繰り上がりをRNNにて学習させる。
```python
import numpy as np
from common import functions
import matplotlib.pyplot as plt

# def d_tanh(x):



# データを用意
# 2進数の桁数
binary_dim = 8
# 最大値 + 1
largest_number = pow(2, binary_dim)
# largest_numberまで2進数を用意
binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)

input_layer_size = 2
hidden_layer_size = 16
output_layer_size = 1

weight_init_std = 1
learning_rate = 0.1

iters_num = 10000
plot_interval = 100

# ウェイト初期化 (バイアスは簡単のため省略)
W_in = weight_init_std * np.random.randn(input_layer_size, hidden_layer_size)
W_out = weight_init_std * np.random.randn(hidden_layer_size, output_layer_size)
W = weight_init_std * np.random.randn(hidden_layer_size, hidden_layer_size)

# Xavier


# He



# 勾配
W_in_grad = np.zeros_like(W_in)
W_out_grad = np.zeros_like(W_out)
W_grad = np.zeros_like(W)

u = np.zeros((hidden_layer_size, binary_dim + 1))
z = np.zeros((hidden_layer_size, binary_dim + 1))
y = np.zeros((output_layer_size, binary_dim))

delta_out = np.zeros((output_layer_size, binary_dim))
delta = np.zeros((hidden_layer_size, binary_dim + 1))

all_losses = []

for i in range(iters_num):
    
    # A, B初期化 (a + b = d)
    a_int = np.random.randint(largest_number/2)
    a_bin = binary[a_int] # binary encoding
    b_int = np.random.randint(largest_number/2)
    b_bin = binary[b_int] # binary encoding
    
    # 正解データ
    d_int = a_int + b_int
    d_bin = binary[d_int]
    
    # 出力バイナリ
    out_bin = np.zeros_like(d_bin)
    
    # 時系列全体の誤差
    all_loss = 0    
    
    # 時系列ループ
    for t in range(binary_dim):
        # 入力値
        X = np.array([a_bin[ - t - 1], b_bin[ - t - 1]]).reshape(1, -1)
        # 時刻tにおける正解データ
        dd = np.array([d_bin[binary_dim - t - 1]])
        
        u[:,t+1] = np.dot(X, W_in) + np.dot(z[:,t].reshape(1, -1), W)
        z[:,t+1] = functions.sigmoid(u[:,t+1])

        y[:,t] = functions.sigmoid(np.dot(z[:,t+1].reshape(1, -1), W_out))


        #誤差
        loss = functions.mean_squared_error(dd, y[:,t])
        
        delta_out[:,t] = functions.d_mean_squared_error(dd, y[:,t]) * functions.d_sigmoid(y[:,t])        
        
        all_loss += loss

        out_bin[binary_dim - t - 1] = np.round(y[:,t])
    
    
    for t in range(binary_dim)[::-1]:
        X = np.array([a_bin[-t-1],b_bin[-t-1]]).reshape(1, -1)        

        delta[:,t] = (np.dot(delta[:,t+1].T, W.T) + np.dot(delta_out[:,t].T, W_out.T)) * functions.d_sigmoid(u[:,t+1])

        # 勾配更新
        W_out_grad += np.dot(z[:,t+1].reshape(-1,1), delta_out[:,t].reshape(-1,1))
        W_grad += np.dot(z[:,t].reshape(-1,1), delta[:,t].reshape(1,-1))
        W_in_grad += np.dot(X.T, delta[:,t].reshape(1,-1))
    
    # 勾配適用
    W_in -= learning_rate * W_in_grad
    W_out -= learning_rate * W_out_grad
    W -= learning_rate * W_grad
    
    W_in_grad *= 0
    W_out_grad *= 0
    W_grad *= 0
    

    if(i % plot_interval == 0):
        all_losses.append(all_loss)        
        print("iters:" + str(i))
        print("Loss:" + str(all_loss))
        print("Pred:" + str(out_bin))
        print("True:" + str(d_bin))
        out_int = 0
        for index,x in enumerate(reversed(out_bin)):
            out_int += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out_int))
        print("------------")

lists = range(0, iters_num, plot_interval)
plt.plot(lists, all_losses, label="loss")
plt.show()
```
{{< figure src="/image/RNN学習結果.png" title="RNN学習結果" class="center" width="470" height="300" >}}


# Section2. LSTM

##### 確認テスト
シグモイド関数を微分した時、入力値が0の時に最大値をとる。その値として正しいものを選択肢から選べ。

##### 回答
(2) 0.25


## 2.0 LSTMの全体像
LSTMの全体像を下記の図に示す。

{{< figure src="/image/LSTM全体像.png" title="LSTM全体像" class="center" width="670" height="300" >}}

## 2.1 CEC

CECとは、中間層の中で「記憶機能」のみを保持するものである。勾配消失・勾配爆発は、勾配が1であれば解決することより、勾配を1とするためにCECが導入され、RNNでは学習する機能と記憶する機能が分離された。(学習の機能については後述）

## 2.2 入力ゲートと出力ゲート

* 入力ゲート：CECに記憶させる情報を調整する  
* 出力ゲート：CECにて記憶された情報をどのように取り出すかを調整する  

入力ゲート、出力ゲートが追加されることで、それぞれのゲートへの入力値の重みを、重み行列で可変可能としている。

## 2.3 忘却ゲート

CECでは過去に記憶した情報が全て残ることから、不要となった過去の情報も記録され続けてしまう。したがって、過去の情報が不要となった時点で忘却する「忘却ゲート」が追加されている。

\begin{eqnarray*}
c(t) = i(t) \times a(t) + f(t) \times c(t-1)
\end{eqnarray*}

ここで、cがCECの状態を表しており、fが忘却レート・iが入力ゲートの割合・aは活性化出力である。


##### 確認テスト1
以下の文章をLSTMに入力し空欄に当てはまる単語を予測したいとする。文中の「とても」という言葉は空欄の予測においてなくなっても影響を及ぼさないと考えられる。このような場合、どのゲートが作用すると考えられるか。「映画おもしろかったね。ところで、とてもお腹が空いたから何か____。」

##### 回答
忘却ゲート

### Section2 実装演習
LSTMを利用したsin波予測モデルの実装演習結果を記載する。
```python
from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM

# パラメータ
in_out_neurons = 1
hidden_neurons = 300
length_of_sequences = 30

model = Sequential()  
model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))  
model.add(Dense(in_out_neurons))  
model.add(Activation("linear"))  
model.compile(loss="mean_squared_error", optimizer="rmsprop")
model.fit(X_train, y_train, batch_size=600, nb_epoch=15, validation_split=0.05)

# 予測
predicted = model.predict(X_test)

# 描写
dataf =  pd.DataFrame(predicted[:200])
dataf.columns = ["predict"]
dataf["input"] = y_test[:200]
dataf.plot()
```

{{< figure src="/image/LSTM結果.png" title="LSTM結果" class="center" width="670" height="300" >}}



# Section3. GRU

Section2で扱ったLSTMでは、入力/出力/忘却ゲートに時刻t,t-1の状態が入力となっており、パラメータ数が非常に多く、計算負荷が高くなってしまうという課題があった。そこで、パラメータ数を大幅に減らし、計算負荷を抑えつつ精度面を担保するGRUが開発された。

{{< figure src="/image/GRU全体像.png" title="GRU全体像" class="center" width="670" height="300" >}}

##### 確認テスト1
LSTMとCECが抱える課題について、それぞれ簡潔に述べよ。

##### 回答
LSTM: 計算負荷が高い  
CEC: 記憶能力しかなく、ニューラルネットワークの学習特性が存在しない（→入力/出力ゲートを追加した）

##### 確認テスト2
LSTMとGRUの違いを簡潔に述べよ。

##### 回答
LSTMでは、入力/出力/忘却ゲートに時刻t,t-1の状態が入力となっており、パラメータ数が非常に多く、計算負荷が高くなってしまうという課題があった。
GRUでは、パラメータ数を大幅に減らし、計算負荷を抑えつつ精度面を担保している。


### Section3 実装演習
GRUを利用した株価予測ソースコードを記載する。GRUはkerasにてmodelが用意されていることから、そのmodelを利用する。
ここでは、実装したソースコードを記載する。
```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense, GRU
import pandas_datareader as pdr
from sklearn.metrics import accuracy_score

df = pdr.get_data_yahoo("AAPL", "2010-11-01", "2020-11-01")
df["Diff"] = df.Close.diff()
df["SMA_2"] = df.Close.rolling(2).mean()
df["Force_Index"] = df.Close * df.Volume
df["y"] = df["Diff"].apply(lambda x: 1 if x > 0 else 0).shift(-1)
df = df.drop(
   ["Open", "High", "Low", "Close", "Volume", "Diff", "Adj Close"],
   axis=1,
).dropna()
# print(df)
X = StandardScaler().fit_transform(df.drop(["y"], axis=1))
y = df["y"].values
X_train, X_test, y_train, y_test = train_test_split(
   X,
   y,
   test_size=0.2,
   shuffle=False,
)
model = Sequential()
model.add(GRU(2, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.fit(X_train[:, :, np.newaxis], y_train, epochs=100)
y_pred = model.predict(X_test[:, :, np.newaxis])
```


# Section4. 双方向RNN

双方向RNN (Bidirectional RNN)では、過去の情報だけでなく、未来の情報を加味することで、精度を向上させるためのモデルである。双方向RNNは、系列の最初のステップから繰り返して順方向に予想することに加えて、系列の最後のステップt=Tからの逆方向の予測も行う、RNNを双方向形に拡張したモデルである。双方向RNNの実用例として、文章の推敲・機械翻訳等が挙げられる。

##### 確認テストなし

##### 自信の考察
「Bidirectional LSTMを用いた誤字脱字検知システム」という論文を読み、双方向RNNによって誤字脱字検知システムの精度がどのように変化するかを確認した。
https://confit.atlas.jp/guide/event-img/jsai2019/3C4-J-9-03/public/pdf?type=in

### Section4. 実装演習

双方向RNNについても、kerasにてmodelが用意されている。
ここでは双方向LSTMについて実装した結果のモデル構築部分を記載する。
```python
#モデルの構築

#Bidirectional（双方向RNN）
from tensorflow.keras.layers import Embedding, Dense, Bidirectional, LSTM

model = keras.Sequential()

#mask_zero = True(0を0埋め用の数値として扱ってくれる)
model.add(Embedding(17781, 64, mask_zero = True))
#LSTM層(return_sequences=True:完全な系列を返す（Flase:最後の出力を返す（LSTMを多層でできる））)
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(5, activation = 'softmax'))
```



# Section5. RNNでの自然言語処理：Seq2Seq

## 5.0 Seqseqとは  
Seq2seqとは、Encoder-Decoderモデルの一種であり、具体的には機械対話や、機械翻訳などで利用されている。Seq2Seqの全体像を下記に示す。
{{< figure src="/image/seq2seq全体像.png" title="seq2seq全体像" class="center" width="670" height="300" >}}

## 5.1 Encoder RNN
ユーザがインプットしたテキストデータを、単語等のトークンに区切って渡す構造のこと。
* Taking: 文章を単語等のトークンごとに分割し、トークンごとのIDに分割する。
* Embedding: IDから、そのトークンを表す分散表現ベクトルに変換する。

(例)
昨日　私　は　刺身　を　食べ　ました　。

## 5.2 Decoder RNN
システムがアウトプットしたデータを、単語等のトークンごとに生成する構造のこと。

##### 確認テスト1

下記の選択肢から、seq2seqについて説明しているものを選べ。  
（1）時刻に関して順方向と逆方向のRNNを構成し、それら2つの中間層表現を特徴量として利用するものである。  
（2）RNNを用いたEncoder-Decoderモデルの一種であり、機械翻訳などのモデルに使われる。  
（3）構文木などの木構造に対して、隣接単語から表現ベクトル（フレーズ）を作るという演算を再帰的に行い（重みは共通）、文全体の表現ベクトルを得るニューラルネットワークである。  
（4）RNNの一種であり、単純なRNNにおいて問題となる勾配消失問題をCECとゲートの概念を導入することで解決したものである

##### 回答
(2)

## 5.3 HRED
Seq2seqの課題としては、一問一答にしか対応できないというものがある。この課題を解決したものがHREDである。HREDでは、過去n-1 個の発話から次の発話を生成する。Seq2seqでは、会話の文脈無視で応答がなされたが、HREDでは前の単語の流れに即して応答されるため、より人間らしい文章が生成される。HREDは、Seq2Seq+ Context RNNで構成されており、過去の発話の履歴を加味した返答をできる。

※Context RNN:   
Encoder のまとめた各文章の系列をまとめて、これまでの会話コンテキスト全体を表すベクトルに変換する構造

## 5.4 VHRED
HREDにVAE(5.5章にて詳細を述べる)の潜在変数の概念を追加したもので、HREDの課題を解決した構造になっている。

##### 確認テスト2
seq2seqとHRED、HREDとVHREDの違いを簡潔に述べよ。

##### 回答
seq2seqでは一問一答にしか対応できないが、HREDでは過去の発話から次の発話を生成するため、文脈を考慮した人間らしい会話のさいげんをすることができる。  
VHREDでは、HREDにVAEの潜在変数の概念を追加したもので、HREDの改良版である。

## 5.5 VAE
### 5.5.1 オートエンコーダ  
オートエンコーダとは教師なし学習の一つであり、学習時の入力データは訓練データのみで教師データを使用しないのが特徴。MNISTの場合、28×28の数字を入れて、同じ画像を出力するニューラルネットワークということになる。入力データから潜在変数zに変換するニューラルネットワークをEncoder、逆に潜在変数zをインプットとして元画像を復元するニューラルネットワークをDecoderとなる。オートエンコーダのメリットとして、次元削減が行えることが挙げられる。

### 5.5.2 VAE
通常のオートエンコーダーの場合、何かしら潜在変数zにデータを押し込めているものの、その構造がどのような状態かわからないという欠点があるが、VAEはこの潜在変数zに確率分布$N(0,1$を仮定したものである。その結果、VAEでは、データを潜在変数zの確率分布という構造に押し込めることを可能とする。

##### 確認テスト3
VAEに関する下記の説明文中の空欄に当てはまる言葉を答えよ。自己符号化器の潜在変数に____を導入したもの。

##### 回答
確率分布

### Section5 実装演習
kerasにて用意されているモデルを利用し、seq2seqの実装を行う。
```python
from keras.models import Model
from keras.layers import Dense, LSTM, Input

n_in = 1  # 入力層のニューロン数
n_mid = 20  # 中間層のニューロン数
n_out = n_in  # 出力層のニューロン数
```
encoderの構築
```python
encoder_input = Input(shape=(n_rnn, n_in))
encoder_lstm = LSTM(n_mid, return_state=True)
encoder_output, encoder_state_h, encoder_state_c = encoder_lstm(encoder_input)
encoder_state = [encoder_state_h, encoder_state_c]
```
decoderの構築
```python
decoder_input = Input(shape=(n_rnn, n_in))
decoder_lstm = LSTM(n_mid, return_sequences=True, return_state=True)
decoder_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_state )
decoder_dense = Dense(n_out, activation='linear')
decoder_output = decoder_dense(decoder_output)
```
modelの構築
```python
model = Model([encoder_input, decoder_input], decoder_output)
```


# Section.6 Word2vec

RNNの課題として、単語のような可変長の文字列をニューラルネットワークに与えることはできない。そのため、固定長形式で単語を表す必要がある。word2vecでは、学習データからボキャブラリを作成する。
word2vecのメリットとしては、大規模データの分散表現の学習が、現実的な計算速度とメモリ量で実現可能にした、という点が挙げられる。

(具体例)  
Ex) I want to eat apples. I like apples.  
→ {apples,eat,I,like,to,want}

Apples を入力とする場合には、入力層には以下のベクトルが入力され、このベクトルをone-hotベクトルという。

1...apples  
0...eat  
0...I  
0...like  
0...to  
...  

word2vecでは、ボキャブラリ×任意の単語ベクトル次元で重み行列が誕生する。

##### 確認テストなし

##### 考察: word2vecの応用例
* fastText  
Word2Vecを考案したトマス・ミコロフが、GoogleからFacebookの人工知能研究所「Facebook AI Research」に移籍し、Word2Vecを発展させる形で生み出したのがfastTextです。その特徴は、圧倒的な単語学習スピードの速さにあります。Facebookは、標準的なCPUを用いた場合でも、10分以内で10億語を学習でき、5分以内で50万もの文を30万のカテゴリーに分類できると公式発表で述べています。

https://fasttext.cc

* Doc2Vec
Doc2Vecは、文章間の類似度やベクトル計算を可能にする手法であり、Paragraph2Vecとも呼ばれます。Googleの研究者であるクオーク・リーが2014年に考案しました。Word2Vecは各単語をベクトルとして処理するのに対し、Doc2Vecでは単語の集合である文章・文書単位でベクトルを割り当てるのが特徴です。たとえば、ニュース記事同士の類似度、レジュメ同士の類似度、本同士の類似度、もちろん人のプロフィールと本の類似度なども算出することができます。テキストで表されているもの同士であれば解析可能です。

* word2vecの活用例：チャットボット
LINEなどで使われるような、くだけた表現や口語混じりの言語は「不自然言語」とも呼ばれます。明治大学の「Word2Vecを用いた顔文字の感情分類」という研究では、不自然言語のひとつでもある「顔文字」に話者が込める感情を分析しています。ツイッターのツイートから、顔文字と感情表現の単語を抽出しベクトル化し、顔文字のベクトルと近い感情表現単語のベクトルを照合することで、顔文字に込められる感情を分析しています。

### Section6 実装演習
pythonのフリーライブラリのgensimを使用してword2vec実装した。以下では、モデル構築部分までを記載する。
```python
from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus('./wiki_wakati.txt')

model = word2vec.Word2Vec(sentences, size=200, min_count=20, window=15)
model.wv.save_word2vec_format("./wiki.vec.pt", binary=True)
```


# Section7 Attention Mechanism

seq2seqの問題は長い文章への対応が困難である。seq2seqでは、2単語でも100単語でも、固定次元ベクトルの中に入力する必要がある。そのため、文章が長くなるほどそのシーケンスの内部表現の次元も大きくなっていく仕組みが必要となり、それで開発されたのがAttention Mechanismである。

(具体例)
{{< figure src="/image/atten具体例.png" title="Attention Mechanismの具体例" class="center" width="670" height="300" >}}

※「a」については日本語で意味を持っておらず、その他の単語との関連度が低いが、「I」については「私」との関連度が高い。

##### 確認テスト
RNNとword2vec、seq2seqとAttentionの違いを簡潔に述べよ。

##### 回答
RNNでは、単語のような可変長の文字列をニューラルネットワークに与えることはできないため、固定長形式で単語を表す必要がある。一方、word2vecでは、学習データからボキャブラリを作成する。
word2vecのメリットとしては、大規模データの分散表現の学習が、現実的な計算速度とメモリ量で実現可能にしたという点が挙げられる。

seq2seqの問題は長い文章への対応が困難である。seq2seqでは、2単語でも100単語でも、固定次元ベクトルの中に入力する必要がある。そのため、文章が長くなるほどそのシーケンスの内部表現の次元も大きくなっていく仕組みが必要となり、それで開発されたのがAttention Mechanismである。

### Section7 実装演習
以下では、Attentionの基本的なソースコードを実装した結果を記載する。なお、入出力は input と memory である。
```python
class SimpleAttention(tf.keras.models.Model):
    '''
    Attention の説明をするための、 Multi-head ではない単純な Attention です
    '''

    def __init__(self, depth: int, *args, **kwargs):
        '''
        コンストラクタです。
        :param depth: 隠れ層及び出力の次元
        '''
        super().__init__(*args, **kwargs)
        self.depth = depth

        self.q_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name='q_dense_layer')
        self.k_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name='k_dense_layer')
        self.v_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name='v_dense_layer')
        self.output_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name='output_dense_layer')

     def call(self, input: tf.Tensor, memory: tf.Tensor) -> tf.Tensor:
        '''
        モデルの実行を行います。
        :param input: query のテンソル
        :param memory: query に情報を与える memory のテンソル
        '''
        q = self.q_dense_layer(input)  # [batch_size, q_length, depth]
        k = self.k_dense_layer(memory)  # [batch_size, m_length, depth]
        v = self.v_dense_layer(memory)

        # ここで q と k の内積を取ることで、query と key の関連度のようなものを計算します。
        logit = tf.matmul(q, k, transpose_b=True)  # [batch_size, q_length, k_length]

        # softmax を取ることで正規化します
        attention_weight = tf.nn.softmax(logit, name='attention_weight')

        # 重みに従って value から情報を引いてきます
        attention_output = tf.matmul(attention_weight, v)  # [batch_size, q_length, depth]
        return self.output_dense_layer(attention_output)
```












