import os
from modeldef import BuildModel

#実行ファイルの場所を作業ディレクトリに設定する
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#実行ディレクトリの取得
Work_Dir = os.path.dirname(os.path.abspath(__file__))

#モデルを読み込み
import modelio
net = BuildModel()
modelio.LoadModel("model.pt", net ,True)
print(net)		#モデルを表示する

device =  'cuda:0' if torch.cuda.is_available() else 'cpu'		#CUDAが使えるなら使う
net.to(device)	#転送


#データセットを準備する
import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
f_mnist_test = FashionMNIST(Work_Dir + "FashionMNIST", train=False, download=True, 
                transform=transforms.ToTensor())

#データーローダーの作成
batch_size = 128
test_loader = torch.utils.data.DataLoader(f_mnist_test, batch_size=batch_size, shuffle=True)

#バッチごとの推論結果を一時格納する配列を宣言する
ys = []
ypreds = []

#ミニバッチ単位で推論を実施、完了までループ
for x, y in test_loader:
    #toメソッド：データを指定の計算デバイスへ転送
    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        _, y_pred = net(x).max(1)

    #結果を配列に追加する
    ys.append(y)
    ypreds.append(y_pred)

#ミニバッチごとにまとめる
#cat関数でテンソルの結合
ys = torch.cat(ys)
ypreds = torch.cat(ypreds)

#予測精度を計算
acc = (ys == ypreds).float().sum() / len(ys)
print("推論結果")
print(acc.item())
