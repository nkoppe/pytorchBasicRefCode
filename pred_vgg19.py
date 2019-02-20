#VGG19モデルをダウンロードしてきて推論を実行するコード

import os

#実行ファイルの場所を作業ディレクトリに設定する
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#実行ディレクトリの取得
Work_Dir = os.path.dirname(os.path.abspath(__file__))

#モデルを読み込み
#学習済みモデルの取得
from torchvision.models import vgg19_bn
net = vgg19_bn(True)
net.eval()      #推論モード
print(net)      #ネットワークの表示

device = "cpu"
net.to(device)


#モデルをシリアライズ
import modelio
modelio.SaveModelWeights(net,"vgg19_bn.pt")
modelio.SaveOnnxModel(net, "model.onnx", (1,224,224))
print("モデル出力完了")


#入力画像を準備する
import torch
from torchvision import transforms
import PIL

#正規化処理の定義
#平均の移動と標準偏差の商を計算
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

#前処理を実施
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),      #PIL形式のデータをテンソルへ変換(チャンネル、高さ、幅の並びになる)
    normalize
])

#画像の読み込みと前処理
img = PIL.Image.open('./test.jpg')
img_tensor = preprocess(img)[:3]        #png対応：アルファチャンネルがあるので3チャンネルに制限
print(img_tensor.shape)

img_tensor.unsqueeze_(0)    #次元を追加
print(img_tensor.shape)

out = net(img_tensor)
print(out.topk(3))
print("処理完了")
