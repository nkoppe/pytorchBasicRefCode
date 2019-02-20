#フォルダから画像データ取得する場合
def GetDataLoader_Image(args):
	pass

#動作検証にFashionMNISTを使う場合
def Get_FashionMNIST(batch_size = 128):
	'''
	FashionMNISTのDataSetを取得し、指定したバッチサイズのDataLoaderを返す
	batch_size: DataLoaderに設定するバッチサイズ

	'''

	#データセットを準備する
	import torch
	from torchvision import transforms
	from torchvision.datasets import FashionMNIST

	f_mnist_train = FashionMNIST(Work_Dir + "FashionMNIST", train=True, download=True, 
					transform=transforms.ToTensor())

	f_mnist_test = FashionMNIST(Work_Dir + "FashionMNIST", train=False, download=True, 
					transform=transforms.ToTensor())

	#データーローダーの作成
	train_loader = torch.utils.data.DataLoader(f_mnist_train, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(f_mnist_test, batch_size=batch_size, shuffle=True)

	#データローダーを返す
	return train_loader, test_loader