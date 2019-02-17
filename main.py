import os


#実行ファイルの場所を作業ディレクトリに設定する
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#実行ディレクトリの取得
Work_Dir = os.path.dirname(os.path.abspath(__file__))

#データセットを準備する
import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST

f_mnist_train = FashionMNIST(Work_Dir + "FashionMNIST", train=True, download=True, 
                transform=transforms.ToTensor())

f_mnist_test = FashionMNIST(Work_Dir + "FashionMNIST", train=False, download=True, 
                transform=transforms.ToTensor())

#データーローダーの作成
batch_size = 128
train_loader = torch.utils.data.DataLoader(f_mnist_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(f_mnist_test, batch_size=batch_size, shuffle=True)


#モデル構築
from modeldef import BuildModel
net = BuildModel()

print(net)      #ネットワーク構造の表示

#自作ヘルパー関数のロード
#評価処理と訓練処理
from trainer import eval_net, train_net

#データをすべて転送する
device_select = "cuda:0"
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