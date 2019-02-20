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
		net = vgg19_bn(False)	#Trueならば学習済みの重みをダウンロードする

		#既にあるモデルファイルから重みをロードする
		import modelio
		modelio.LoadModel("vgg19_bn.pt", net, False)

		#畳み込み部は流用
		self.features = net.features		#モデルを定義（たたみ込み部)

		#畳み込み部の出力サイズを調べる
		size = (3, 224, 224)		#入力画像のテンソル形状
		test_input = torch.ones(1,size[0],size[1],size[2])	#ダミーの入力データ
		temp = self.features(test_input)			#畳み込みする
		temp = temp.view(temp.size()[0], -1)		#Flattenをかける
		conv_output_size = temp.size()[-1]		#出力テンソルサイズを取得

		#クラス分類を定義
		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(conv_output_size,512),
			torch.nn.ReLU(),
			torch.nn.BatchNorm1d(512),
			torch.nn.Dropout(0.25),
			torch.nn.Linear(512,101)
		)
		
	def forward(self, x):
		x = self.features(x)				#特徴抽出
		x = x.view(x.size()[0], -1)		#Flatten
		x = self.classifier(x)			#分類
		return torch.nn.Softmax(x)


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
