import os


#実行ファイルの場所を作業ディレクトリに設定する
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#実行ディレクトリの取得
Work_Dir = os.path.dirname(os.path.abspath(__file__))

#DatasetとDataLoaderの準備
import load_dataset
batch_size = 128
train_loader , test_loader = load_dataset.Get_FashionMNIST(batch_size)


#モデル構築
from modeldef import VGG19custom
net = VGG19custom()

print(net)      #ネットワーク構造の表示

#自作ヘルパー関数のロード
#評価処理と訓練処理
from trainer import eval_net, train_net

#データをすべて転送する
import torch
device_select = 'cuda' if torch.cuda.is_available() else 'cpu'		#CUDAが使えるなら使う
n_epoch = 5
net.to(device_select)

#訓練実施
train_net(net, train_loader, test_loader, n_iter=n_epoch, device=device_select)
print("モデル訓練完了")

#モデルをシリアライズ
import modelio
modelio.SaveModelWeights(net,"model.pth")
modelio.SaveOnnxModel(net, "model.onnx", (1,28,28))
print("モデル出力完了")