import tqdm
import torch

#ヘルパー関数
def eval_net(net, data_loder, device="cpu"):
    #推論モードにする(DropoutやBatchNormを無効化する)
    net.eval()

    #バッチごとの推論結果を一時格納する配列を宣言する
    ys = []
    ypreds = []

    #ミニバッチ単位で推論を実施、完了までループ
    for x, y in data_loder:
        #toメソッド：データを指定の計算デバイスへ転送
        x = x.to(device)
        y = y.to(device)

        #確率が最大のクラスを予測
        #確率が最大のインデックスを取得する
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
    return acc.item()

#訓練用ヘルパー関数
def train_net(net, train_loader, test_loader,
                optimizer_cls=torch.optim.Adam,
                loss_fn = torch.nn.CrossEntropyLoss(),
                n_iter=10, device="cpu"):
    
    #ログを格納するための配列を宣言する
    train_losses = []
    train_acc = []
    val_acc = []

    #パラメータを渡してオプティマイザの初期化
    optimizer = optimizer_cls(net.parameters())

    #イテレーション回数(エポック数)だけループ
    for epoch in range(n_iter):
        #変数の初期化
        running_loss = 0.0

        #ネットワークを訓練モードにする
        net.train()

        #変数を初期化する
        n = 0
        n_acc = 0

        #時間がかかるのでプログレスバーを表示する
        for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader),
                    total=len(train_loader)):
            #ローダーから取得したデータを指定デバイスに転送する
            xx = xx.to(device)
            yy = yy.to(device)

            #まずは推論値の計算
            h = net(xx)

            #推論結果をラベルと突き合わせて損失を計算する
            loss = loss_fn(h, yy)
            optimizer.zero_grad()
            
            #微分の実行
            loss.backward()

            #勾配を更新する
            optimizer.step()
            
            #バッチの損失をエポック通算の損失へ加算
            running_loss += loss.item()

            #完了したデータ数を加算（進捗状況表示用）
            n += len(xx)

            #推論結果から最大値を示したインデックスを取得する
            _, y_pred = h.max(1)

            #正解数を加算する
            n_acc += (yy == y_pred).float().sum().item()
        
        #損失のログを取る
        train_losses.append(running_loss / i)

        #訓練データの予測精度
        train_acc.append(n_acc / n)
        
        #検証データの予測精度
        val_acc.append(eval_net(net, test_loader, device))

        #このエポックの結果を表示
        print(epoch, train_losses[-1], train_acc[-1], val_acc[-1], flush=True)