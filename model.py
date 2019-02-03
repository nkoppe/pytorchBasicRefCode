import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST

def BuildModel():
    #(N, C, H, W)形式のTensorを(N, C*H*W)に引き延ばす層
    #畳み込み層の出力をMLPに渡す際に必要
    class FlattenLayer(torch.nn.Module):
        def forward(self, x):
            sizes = x.size()
            return x.view(sizes[0], -1)

    #畳み込みネットワークの定義
    Network_Convolution = torch.nn.Sequential(
        torch.nn.Conv2d(1,32,5),
        torch.nn.MaxPool2d(2),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(32),
        torch.nn.Dropout2d(0.25),
        torch.nn.Conv2d(32,64,5),
        torch.nn.MaxPool2d(2),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(64),
        torch.nn.Dropout2d(0.25),
        FlattenLayer()
    )

    #Linear（全結合層）を使うときは、入力テンソルの形状を指定する必要あり
    #畳み込み部に適当なテンソルを入力して出力サイズを取得する
    test_input = torch.ones(1,1,28,28)
    conv_output_size = Network_Convolution(test_input).size()[-1]

    #2層のMLP
    Network_MLP = torch.nn.Sequential(
        torch.nn.Linear(conv_output_size,200),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(200),
        torch.nn.Dropout(0.25),
        torch.nn.Linear(200,10)
    )

    #最終的なモデル
    net = torch.nn.Sequential(Network_Convolution, Network_MLP)
    return net

