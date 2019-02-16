import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST

class DNNModel:
    '''
    PyTorchのDNNモデルを定義・制御
    '''

    size = None   #モデルサイズの保持
    network = None

    
    #コンストラクタ
    def __init__(self):
        '''
        コンストラクタ
        '''
        self.network = self.BuildModel()    #ネットワークを構築する

    def BuildModel(self):
        '''
        モデルを定義・構築する
        '''
        size = (1,28,28)
        self.size = size
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
        test_input = torch.ones(1,size[0],size[1],size[2])
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
        
        #モデルをメンバ変数へ格納する
        return net


    def SaveModel(self, filename="net.pt", net=None):
        '''
        モデルの保存を
        PyTorchのファイル拡張子の慣例は.pt/.pth
        net: 保存するネットワーク
        filename: ファイル名
        evalMode: 読み込み後に推論モードにするかどうか
        '''

        #ネットワークが指定されていない場合
        if net is None:
            net = self.network  #ネットワークを指定していない場合はクラスのネットワークを代入

        #モデルをCPU側へ転送
        #GPUに乗せたまま保存するとGPUが無いとロードできなくなる
        net.cpu()

        #ネットワークの重みを保存する
        torch.save(net.state_dict(), filename)
        #ネットワーク情報が無いため注意
        #torch.save(net)でネットワークも出力可能（公式は非推奨）



    def LoadModel(self, filename = "net.pt", evalMode=False):
        '''
        モデルの読み込みを行う関数
        fimename: ファイル名
        '''
        net = self.network  #ネットワークの読み込み

        net.load_state_dict(torch.load(filename))   #重みのロード

        #推論モードへの変更
        if evalMode:
            net.eval()


    def SaveOnnxModel(self, filename="net.onnx"):
        '''
        ONNXモデルの保存を行う
        net: 保存するネットワーク
        filename: ファイル名
        size: 入力配列のサイズ
        '''

        size = self.size

        #ネットワークを取得
        net = self.network

        #必要なモジュールをインポート
        import torch.onnx

        #モデルを保存前に推論モードに変更する
        net.eval()
    
        #ダミーデータの作成
        #動的計算
        dummydata = torch.empty((1, size[0], size[1], size[2]), dtype=torch.float32)

        #ONNXファイルへネットワークを出力する
        torch.onnx.export(net, dummydata, filename)

