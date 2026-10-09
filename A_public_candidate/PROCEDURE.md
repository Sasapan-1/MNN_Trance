# A 手順書: 学習から真理値表ネットワーク評価まで

この手順は、現在の A フォルダだけで実行する範囲を示す。A は公開候補であり、公開承認済みのパッケージではない。SystemVerilog 用の MEM/SVH 生成と RTL シミュレーションは B の手順書を参照する。

## 1. 実行環境

PowerShell でプロジェクトのルートから A に移動する。

~~~powershell
Set-Location .\trancerate\A_public_candidate
python --version
~~~

Python 3.11 と requirements.txt のライブラリを用いる。互換性のある隔離環境を用意した場合は、そこで次を実行する。観測済みのバージョンは environment_observed.md にあるが、requirements.txt は厳密なロックファイルではない。

~~~powershell
python -m pip install -r .\requirements.txt
~~~

実行入口は source\model_core.py の6層・11層定義と source\mrp_core.py の真理値表・MRP中核関数を使用する。旧一体型ソースは同梱しない。学習済みチェックポイントと保存済み入出力層の読込・推論は次で確認できる。

~~~powershell
python -m unittest discover -s .\tests -v
~~~

MNIST は A\data に無ければ torchvision が取得する。ネットワークから取得できない環境では、MNIST を別途用意し、後述のコマンドに --data-dir を指定する。GPU が不要なら --device cpu を指定する。

## 2. 同梱チェックポイントの確認

対象は 6 層 BMNN と 11 層 Deep-BMNN であり、いずれも最終 700 エポックのチェックポイントを同梱している。

| モデル指定 | チェックポイント |
| --- | --- |
| mnist6_binary | checkpoints\bmnn_6_epoch700\MNIST_DeepBMNNmodel1v5_6_r_para.pth |
| mnist11_forward | checkpoints\deep_bmnn_11_epoch700\MNIST_DeepBMNNmodel1v5_11_para.pth |

NN 単体の推論・精度確認だけを行う場合は次を実行する。結果は runs\selection.json に保存される。

~~~powershell
python .\source\mnist_mrp_pipeline.py select --models mnist6_binary mnist11_forward --out .\runs\selection.json --device cpu
~~~

## 3. まず少数サンプルで動作確認

この確認は処理が通ることを見るもので、論文用精度の測定ではない。

~~~powershell
python .\source\mnist_mrp_pipeline.py prepare --models mnist6_binary mnist11_forward --max-samples 1 --batch-size 1 --device cpu --out-dir .\runs\smoke
~~~

各モデルの runs\smoke\<モデル指定>\truth_table\validation.json で validation_errors が空配列、comparisons\mrp_end_raw_vs_truth.json で all_equal が true であることを確認する。実行時には、モデルの読込、NN 推論、各ノードの真理値表生成、真理値表グラフ作成、MRP への配置と中立 JSON 出力、真理値表ネットワーク推論、NN とのビット比較が順に行われる。

## 4. MNIST テスト全件の評価

少数サンプルの指定を外して実行する。既定では MNIST テスト分割を使用し、6 層は right_to_left、11 層は fold の MRP 配置となる。別条件を試す場合だけ --flow-direction を指定する。

~~~powershell
python .\source\mnist_mrp_pipeline.py prepare --models mnist6_binary mnist11_forward --device cpu --out-dir .\runs\neutral
~~~

各モデルの出力を確認する。

| 出力 | 内容 |
| --- | --- |
| raw\metrics.json | 元の NN の分類精度 |
| truth\metrics.json | 真理値表ネットワークの分類精度 |
| comparisons\mrp_end_raw_vs_truth.json | MRP 終端ビットの NN/真理値表一致数 |
| truth_table\neutral_mrp_network.json | MRP 配置済みの中立ネットワーク |
| truth_table\validation.json | 中立ネットワークの形式検証 |
| truth_table\export_io_layers\output_layer.pt | MRP 外部の最終分類層 |
| inputs\input_vectors.pt、inputs\labels.txt | B へ渡す固定入力とラベル |
| manifest.json | 使用モデル、チェックポイント、配置、評価の記録 |

PowerShell では、例えば 6 層の結果を次で確認できる。

~~~powershell
Get-Content .\runs\neutral\mnist6_binary\raw\metrics.json
Get-Content .\runs\neutral\mnist6_binary\truth\metrics.json
Get-Content .\runs\neutral\mnist6_binary\comparisons\mrp_end_raw_vs_truth.json
Get-Content .\runs\neutral\mnist6_binary\truth_table\validation.json
~~~

全件実行の確認基準は、各モデルで sample_count が 10000、終端 8 ビットの total_bits が 80000、all_equal が true、validation_errors が空であること。数値は今回の再実行結果として確認してから記録する。同梱の過去結果は 6 層が 91.42%、11 層が 92.29% だが、6 層チェックポイント内の学習履歴は最終 91.48% であり、この差を混同しない。

## 5. 新たに学習する場合

学習入口は source\train_mnist.py。--epochs は追加エポック数ではなく、到達したい総エポック数である。出力先はモデルごとに分ける。

~~~powershell
python .\source\train_mnist.py --model bmnn_6 --epochs 700 --out-dir .\training_runs\bmnn_6
python .\source\train_mnist.py --model deep_bmnn_11 --epochs 700 --out-dir .\training_runs\deep_bmnn_11
~~~

既存チェックポイントから継続するときは --resume にそのファイルを指定し、--epochs に開始エポックより大きい総数を指定する。新規学習の重みや精度は、乱数状態などを揃えない限り同梱チェックポイントと一致するとは限らない。

新規学習したモデルを評価するときは、--checkpoint をモデルごとに指定して別々の出力先へ prepare を実行する。

~~~powershell
python .\source\mnist_mrp_pipeline.py prepare --models mnist6_binary --checkpoint .\training_runs\bmnn_6\MNIST_DeepBMNNmodel1v5_6_r_para.pth --out-dir .\runs\trained6
python .\source\mnist_mrp_pipeline.py prepare --models mnist11_forward --checkpoint .\training_runs\deep_bmnn_11\MNIST_DeepBMNNmodel1v5_11_para.pth --out-dir .\runs\trained11
~~~

## 6. B への受け渡し

標準の全件評価から B に渡すディレクトリは runs\neutral\mnist6_binary と runs\neutral\mnist11_forward。各ディレクトリの中立 JSON、入力テンソル、ラベル、保存された出力層を同じ実行の組として扱う。A では MEM/SVH を生成しない。続きは B_private_candidate\PROCEDURE.md を参照する。

generated\ にある中立 JSON は収集済みの歴史的成果物である。新規実行の受け渡しには runs\neutral 以下を使用する。
