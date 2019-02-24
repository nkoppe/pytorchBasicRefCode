import os


#実行ファイルの場所を作業ディレクトリに設定する
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#実行ディレクトリの取得
Work_Dir = os.path.dirname(os.path.abspath(__file__))

#DatasetとDataLoaderの準備
import load_dataset
batch_size = 64
from torchvision import transforms
#正規化処理を定義
tf_nrm = transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
tf = transforms.Compose([
	transforms.Resize((224,224)),		#(h,w)で指定
	transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
	transforms.ToTensor(),      #PIL形式のデータをテンソルへ変換(チャンネル、高さ、幅の並びになる),値は[0,1]の範囲になる
	transforms.Lambda(lambda x : 1.0 - x ),		#ラムダ式による処理の定義
	tf_nrm])	

#ImageFolderでデータセットを読み込み、分割してデータローダーを作る
from torchvision.datasets import ImageFolder
train_loader, test_loader = load_dataset.GetDataLoader_withSplit(ImageFolder('./food-101/images',tf), 0.2, batch_size)
print("データーローダー準備完了")

#モデル構築
from modeldef import VGG19custom
net = VGG19custom()

print(net)      #ネットワーク構造の表示

#自作ヘルパー関数のロード
#評価処理と訓練処理
from trainer import eval_net, train_net

#データをすべて転送する
import torch
#device_select = 'cpu'		#デバッグ用
device_select = 'cuda' if torch.cuda.is_available() else 'cpu'		#CUDAが使えるなら使う
n_epoch = 5
net.to(device_select)

#訓練実施
train_net(net, train_loader, test_loader, n_iter=n_epoch, device=device_select)
print("モデル訓練完了")

#モデルをシリアライズ
import modelio
modelio.SaveModelWeights(net,"model.pth")
modelio.SaveOnnxModel(net, "model.onnx", (3,224,224))
print("モデル出力完了")