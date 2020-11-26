## 必要ライブラリ群のインストール 
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
 
## 学習に用いるネットワークをクラスで定義
## 全結合(fc1) -> sigmoid -> 全結合(fc2) -> クロスエントロピーのネットワーク構成
## 今回は簡単のため全結合層しか用いていませんが画像の学習の際はほぼ100%、CNNという層をネットワークに使います。
class DLNetwork(torch.nn.Module):
    def __init__(self):
        super(DLNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 1000)
        self.fc2 = torch.nn.Linear(1000, 10)
 
    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
 
        return f.log_softmax(x, dim=1)
 

## MNISTデータセットという、手書き数字のたくさん入った画像データセットが公開されているのでそれを利用します
def load_MNIST(batch=128, intensity=1.0):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data',
                       train=True,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x * intensity)
                       ])),
        batch_size=batch,
        shuffle=True)
 
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data',
                       train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x * intensity)
                       ])),
        batch_size=batch,
        shuffle=True)
 
    return {'train': train_loader, 'test': test_loader}
 

## 本ファイル(train.py)を`python train.py`で実行した時に最初に呼び出される特殊な関数です
if __name__ == '__main__':
    # 学習回数
    epoch = 20
 
    # ネットワークを構築
    model: torch.nn.Module = DLNetwork()
 
    # MNISTのデータローダを取得
    loaders = load_MNIST()
 
    # 損失関数の最小点を探すアルゴリズムを指定します。Adamは現在メジャーです。
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    # GPUを使うための設定です。PyTorchではGPUを使用したい場合、明示的指定が必要です。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    for e in range(epoch):
        """訓練（training）セッション"""
        loss = None
        # 学習開始
        model.train(True)  # 引数は省略可能
        for i, (data, target) in enumerate(loaders['train']):
            data = data.view(-1, 28*28)  # 全結合層のノードに落とすため28*28の２次元行列を28*28個の要素からなるベクタに変換する
            data.to(device)  # GPU演算を行うためにdataの内容をVRAMにアサインするイメージ(厳密には異なる)
 
            optimizer.zero_grad()

            output = model(data)  # ネットワークの出力値
            output.to(device)  # GPU演算を行うためにoutputの内容をVRAMにアサインするイメージ(厳密には異なる)

            loss = f.nll_loss(output, target)  # 損失関数の出力
            loss.backward()
            optimizer.step()  # 極値探索（最小値探し）を次のステップへ
 
            if i % 10 == 0:
                print('Training log: {} epoch ({} / 60000 train. data). Loss: {}'.format(e+1, (i+1)*128, loss.item()))
 

        """ テスト（testing）セッション"""
        # テスト：テスト用にmnistから分離したデータに対して学習時同様の処理をかけることで、学習処理部の検証を行うことができる。
        model.eval()  # または model.train(False) でも良い
        test_loss = 0
        correct = 0
 
        with torch.no_grad():
            for data, target in loaders['test']:
                data = data.view(-1, 28 * 28)
                data.to(device)

                output = model(data)
                output.to(device)

                test_loss += f.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
 
        test_loss /= 10000
 
        print('Test loss (avg): {}, Accuracy: {}'.format(test_loss, correct / 10000))

