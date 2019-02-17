import os
import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST

#モデル構築
#より一般的な方法
class VGG19custom(torch.nn.Module):
	def __init__(self):
		super(VGG19custom, self).__init__()		#基底クラスのコンストラクタを実行する
		
		#学習済みモデルの取得
		from torchvision.models import vgg19_bn
		net = vgg19_bn(True)

		#たたみ込み部は流用
		self.features = net.features		#モデルを定義（たたみ込み部)

		
		self.fc1 = nn.Linear(4, 100)
		self.fc2 = nn.Linear(100, 50)
		self.fc3 = nn.Linear(50, 3)

	def forward(self, x):
		x = self.features(x)
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.log_softmax(x, dim = 1)


#モデル構築
#Sequentialモデルでのみ可能な簡易な方法
def BuildModel():
	'''
	モデルを定義・構築する
	'''
	size = (1,28,28)

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
