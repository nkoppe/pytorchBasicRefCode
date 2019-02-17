import torch
import numpy as np

def SaveModelWeights(net, filename):
	'''
	モデルの重みを保存
	PyTorchのファイル拡張子の慣例は.pt/.pth
	net: 保存するネットワーク
	filename: ファイル名
	'''

	#モデルをCPU側へ転送
	#GPUに乗せたまま保存するとGPUが無いとロードできなくなる
	net.cpu()

	#ネットワークの重みを保存する
	torch.save(net.state_dict(), filename)
	#ネットワーク情報が無いため注意
	#torch.save(net)でネットワークも出力可能（公式は非推奨）


def SaveModelData(net, filename):
	'''
	モデルの重みと構造、環境情報を保存（非推奨）
	PyTorchのファイル拡張子の慣例は.pt/.pth
	net: 保存するネットワーク
	filename: ファイル名
	
	※このメソッドはファイルに環境情報を含んでしまうことにより
	　別環境での再利用性が悪化するため非推奨
	'''

	#モデルをCPU側へ転送
	#GPUに乗せたまま保存するとGPUが無いとロードできなくなる
	net.cpu()

	#ネットワークの重みを保存する
	torch.save(net, filename)

def LoadModel(filename, net ,evalMode=True):
	'''
	モデルの読み込みを行う(重みの読み込み)
	fimename: ファイル名
	modelClass: 重みを適用するモデル
	evalMode: 推論モードに変更する
	'''
	#重みをロード
	net.load_state_dict(torch.load(filename))

	#推論モードへの変更
	if evalMode:
		net.eval()

	return net


def SaveOnnxModel(net, filename, size):
	'''
	ONNXモデルの保存を行う
	net: 保存するネットワーク
	filename: ファイル名
	size: 入力配列のサイズ(注バッチの次元は含まない 例:画像は(3,224,224)などを入力)
	'''

	#必要なモジュールをインポート
	import torch.onnx

	#モデルを保存前に推論モードに変更する
	net.eval()

	#軸を追加(ここでバッチサイズの次元を追加する)
	size = np.insert(size, 0, 1)

	#ダミーデータの作成
	#動的計算
	dummydata = torch.empty( tuple(size.tolist()) , dtype=torch.float32)

	#ONNXファイルへネットワークを出力する
	torch.onnx.export(net, dummydata, filename)

