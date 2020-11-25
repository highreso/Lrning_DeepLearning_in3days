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
        super(MyNet, self).__init__()
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
    net: torch.nn.Module = DLNetwork()
 
    # MNISTのデータローダを取得
    loaders = load_MNIST()
 
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
 
    for e in range(epoch):
 
        """訓練（training）セッション""""
        loss = None
        # 学習開始
        net.train(True)  # 引数は省略可能
        for i, (data, target) in enumerate(loaders['train']):
            # 全結合のみのネットワークでは入力を1次元に
            data = data.view(-1, 28*28)  # 全結合層のノードに落とすため28*28の２次元行列を28*28個の要素からなるベクタに変換する
 
            optimizer.zero_grad()
            output = net(data)
            loss = f.nll_loss(output, target)
            loss.backward()
            optimizer.step()
 
            if i % 10 == 0:
                print('Training log: {} epoch ({} / 60000 train. data). Loss: {}'.format(e+1, (i+1)*128, loss.item()))
 

        """ テスト（testing）セッション"""
        # テスト：テスト用にmnistから分離したデータに対して学習時同様の処理をかけることで、学習処理部の検証を行うことができる。
        net.eval()  # または net.train(False) でも良い
        test_loss = 0
        correct = 0
 
        with torch.no_grad():
            for data, target in loaders['test']:
                data = data.view(-1, 28 * 28)
                output = net(data)
                test_loss += f.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
 
        test_loss /= 10000
 
        print('Test loss (avg): {}, Accuracy: {}'.format(test_loss, correct / 10000))

