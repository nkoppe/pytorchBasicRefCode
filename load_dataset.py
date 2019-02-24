#データセット取得　画像ファイル名からラベルを取得する
def GetDataSet_ImageRegression(datadir, batchsize, transform, shuffle = True, random_seed = 42):
	'''
	回帰問題向けデータセット作成
	datadir: 画像データが格納されたフォルダ
	batchsize: バッチサイズ
	'''
	#必要なモジュールをインポート
	import torch
	from torchvision.datasets import DatasetFolder

	dataset = ImageFolder(datadir,transform)
	return dataset
	

#データを分割する関数
def GetDataLoader_withSplit(dataset, validation_split, batch_size, shuffle = True, random_seed = 42):
	'''
	データセットの分割処理
	dataset: 分割するデータの元データ
	splitrate: 確認データの比率(0.0～1.0)
	shuffle: シャッフル処理の有無
	random_seed: ランダムシード
	'''
	import torch.utils
	import numpy as np

	# Creating data indices for training and validation splits:
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	split = int(np.floor(validation_split * dataset_size))
	if shuffle :
		np.random.seed(random_seed)
		np.random.shuffle(indices)
	train_indices, val_indices = indices[split:], indices[:split]
	
	#DataSamplerとDataLoaderを作成する
	train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
	valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

	train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
											   sampler=train_sampler)
	validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
													sampler=valid_sampler)

	return train_loader, validation_loader


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