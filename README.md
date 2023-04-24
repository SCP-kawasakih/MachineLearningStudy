# 1章　機械学習の現状

### Gerbage in, gerbage out
- モデルの質も大事だが訓練するデータの質はもっと大事
- ノイズや無関係な特徴量などの見極め
### 検証セットの意義
- モデルの評価を行うのがテストセットのみになると、そのテストセットに合わせて何度もモデルやハイパーパラメーターを調整することになり、そのテストセットだけに合ったモデルが完成する。それは精度の高いモデルとはいえない。
    - ハイパーパラメータ
        - 正則化の程度を決めるもので、訓練の前に設定する。過学習を防いでくれるが、大きすぎると鈍感な学習になる。

# 2章　エンドツーエンドの機械学習プロジェクト

### 設計の前に意識する
- 何のための予測タスク？
    - これによってアルゴリズムの選定、評価指標、求める精度などが変わる
    - 「回帰を使って細かい値まで予測できるようにしたのに実際はカテゴリーに分ける（分類）だけでよかった」は悲しい
- その問題に対して今はどうしてる？
    - 性能の比較対象になる
        - 性能として最低限ここは超えないといけない
### 層化抽出法
- テストセットを作成する際にデータの中身が、重要と思われる属性の割合が現実の割合に近くなるようにする
- 例えば男女比が2:3の組織のメンバーデータセットから何かのタスクを行うモデルを設計する場合、テストセットの男女比も2:3になるようにした方がいい、という考え方。
### 属性選びが大事
- 必要・不必要を見分けるのももちろん、属性と属性を掛け合わせて新たな属性を作るのもかなり有効
- 例：「人口」と「世帯数」から「１世帯当たりの人数」など
- 標準相関係数を計算
    - ~~~python
        corr_matrix = housing.corr()
- 散布図行列の表示
    - ~~~python
        from pandas.plotting import scatter_matrix
        # 表示したい属性
        attributes = ["median_house_valie", "median_income", "total_rooms", "housing_median_age"]

        scatter_matrix(housing[attributes], figsize=(12, 8))
### データの前処理
- 欠損値処理
    - ~~~python
        from sklearn.impute import SimpleImputer
        # その属性の中央値で埋める設定に
        imputer = SimpleImputer(strategy="median")
        # 中央値はデータが数値であることが前提なのでテキスト型の属性列をどけておく
        housing_num = housing.drop("ocean_proximity", axis=1)
        # 訓練データにimputerインスタンスを適合（中央値を計算）
        imputer.fit(housing_num)
        # 欠損値に中央値をセットしていく
        X = imputer.transform(housing_num)
        # なぜかNumpyになっているのでPandasに変換しとく
        housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
- カテゴリ属性の処理
    - 先ほどどけたテキスト型の属性はただのテキストではなくカテゴリ属性といえそうなので、操作しやすい数値に変換する
    - ~~~python
        from sklearn.preprocessing import OrdinalEncoder

        # テキスト型のカテゴリ属性ocean_proximityを抽出
        housing_cat = housing[["ocean_proximity"]]       
        ordinal_encoder = OrdinalEncoder()
        # OrdinalEncoderを適合・データを変換
        housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    - それぞれのカテゴリは完全に自立したものなのでカテゴリごとにバイナリ属性を作る
    - ~~~python
        from sklearn.preprocessing import OneHotEncoder
        cat_encoder = OneHotEncoder()
        # OneHotEncoderを適合・データを変換
        housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
        # なぜかSciPyの疎行列なのでNumPyに変換
        housing_cat_1hot.toarray()
- カスタム変換器
    - BaseEstimatorを継承
        - よくわからないが便利らしい。勉強が必要
    - TransformerMixinも継承
        - これによりfit_transformメソッドが使えるようになるとのこと
    - ~~~python
        # 二つの属性を組み合わせて新たな属性を追加するための変換器
        # １世帯当たりの部屋数、１世帯当たりの人数、必要であれば寝室の割合を新たな属性として追加
        from sklearn.base import BaseEstimator, TransformerMixin
        # 部屋数、寝室数、人口、世帯数の列番号
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        
        class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
            # デフォルトで寝室の割合を追加するようにしておく
            # 属性の効果を調べられるようにするためにフラグで管理
            def __init__(self, add_bedrooms_per_room = True):
                self.add_bedrooms_per_room = add_bedrooms_per_room
            def fit(self, X, y=None):
                return self
            def transform(self, X):
                # 見ての通り、属性を作っていく
                rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
                population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

        # 実行例
        attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
        housing_extra_attribs = attr_adder.transform(housing.values)
- パイプライン
    - scilit-learnにはこれら複数のデータ変換処理をまとめて行うクラスがある
    - ~~~python
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        num_pipeline = Pipeline([
            # （名前、変換器もしくは推定器）のペア
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler())
        ])
        # 最後の変換のメソッドを呼び出すと他の変換は自動的にfit_transformされる
        housing_num_tr = num_pipeline.fit_transform(housing_num)
- ColunmTransformer
    - テキストのカテゴリ列と数値の列を同時に変換できる
    - ~~~python
        from sklearn.compose import ColumnTransformer
        # カテゴリ列と数値列を分けて用意しておく必要はある
        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]
        # 数値列には先程のパイプライン、カテゴリ列にはワンホットエンコーダー
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])
        # 変換実行
        housing_prepared = full_pipeline.fit_transform(housing)
### モデルの訓練と評価
- K分割交差検証
    - 訓練セットを複数のサブセット（fold）に無作為に分割してそのうち9個で訓練を行い、1個を評価用にする。
    - ~~~python
        # 決定木モデルの評価
        from sklearn.model_selection import cross_val_score
        # モデル、訓練セット、正解ラベル、スコアリング形式（？）、fold数を引数にとる
        score = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_aquared_error", cv=10)
        tree_rmse_scores = np.sqrt(-scores)

### モデルの微調整
- グリッドサーチ
    - 最適なハイパーパラメータを求めてくれる
    - ~~~python
        # RandomForestRegressorモデルのハイパーパラメータ値の最高値を探す
        from sklearn.model_selection import GridSearchCV
        # ハイパーパラメータの種類など細かいことは７章で勉強
        param_grid = [
            {'n_estimation': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
            {'bootstrap': [Flase], 'n_estimation': [3, 10], 'max_features': [2, 3, 4]}
        ]
        forest_reg = RandomForestRegressor()
        # 対象のモデルやハイパーパラメータのパターンなどの情報をぶち込む
        grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                                    scoring='neg_mean_aquared_error,
                                    return_train_score'=True)
        # 訓練データに適合させる
        grid_search.fit(housing_prepared, housing_labels)
        # 最高のパラメータの組み合わせを出力
        grid_search.best_params_

        # 最良の推定器を得る（これ一番大事やろ）
        grid_search.best_estimator_
        # 評価のスコアを出力
        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)
# 3章　分類

### 性能指標
- 歪んだデータセットの場合、正解率が当てにならない場合が多い
    - 例えば0〜9の数字でラベル付けされた一様に分布したデータがあり、そのラベルが9かどうかを予測するタスクがあるとして、もし全て「9じゃない」と予測すれば正解率は90％になるがこのモデルの精度が悪いことは確実
    - 適合率
        - ◯と予測したもののうち、本当に◯だったものの割合
    - 再現率
        - ◯であるもののうち、ちゃんと◯だと予測できたものの割合
    - 個々の予測結果を返すK分割交差検証
        - ~~~python
            from sklearn.model_selection import cross_predict
            # 分類モデル、訓練セット、正解ラベル、fold数を引数にとる
            y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
            # method引数を追加すれば予測結果の形式を単なるラベルから具体的なスコアなどに変更できる
    - 混同行列
        - 実際のラベルと予測されたラベルからなる行列で、各要素にはそのデータの合計が入る
        - ~~~python
            from sklearn.metrix import confusion_matrix
            # 正解ラベル、個々の予測結果を引数にとる
            confusion_matrix(y_train_5, y_train_pred)
    - 適合率と再現率はトレードオフの関係にあり、どちらかが上がればもう一方は下がる
        - しきい値を操作することで調節できる
- 最適なしきい値
    - ~~~python
        # 全インスタンスのスコアを返す（メソッドを指定することで予測値ではなく他の値を返せる）
        y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
        
        from sklearn.metrix import precision_recall_curve
        # 正解ラベル、個々のスコアを引数にとり、適合率、再現率、しきい値を返す
        precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
        # 90%の適合率を目指す
        # しきい値が上がれば適合率も上がる。適合率が90％を超える最小のしきい値を求める
        # argmaxは最大値を持つ要素のインデックスを返す関数であるが、TrueやFalseのようなブール値を要素として持つ配列に対して使用する場合には、Trueを表す値が最初に出現するインデックスを返す
        threshold_90_precision = thresholds[np.argmax(presicions >= 0.90)]
    - 上記のようなしきい値・適合率・再現率はMatplotlibを使ってグラフにして視覚化すべし
- ROC曲線
    - 真陽性率（適合率） / 偽陽性率
        - 適合率 / 再現率　は「PR曲線」
    - 偽陽性率がどんな値をとろうとも（たとえ低くても）真陽性率は高い方がいい
        - 真陽性率=1.0で横一直線のグラフが理想（PR曲線も同じだが増減が逆）
    - AUC
        - 曲線の下側の面積を指標としたもの
        - 1.0が最大値であり理想
    - 珍しいクラスがあったり、特定のクラスが気になるなどの場合はPR曲線、それ以外の場合はROC曲線（目安の基準）
        - PR曲線では適合率と再現率が共に高くないと理想の形（適合率=1.0の横一直線）に近づかないため
        
### 他クラス分類

- OVA法
    - 他クラス分類を二項分類の連続で行う方法
    - 例：0とそれ以外、1とそれ以外…
- OVA法
    - 同じく他クラス分類を二項分類の連続で行う方法
    - 例：0と1、0と2、0と3…

# 4章　モデルの訓練

### 線形回帰
- 二乗平均誤差を０にするパラメーターを求める
    - 二乗平均誤差=0から導出される正規方程式で求められる
- 大量の特徴量が存在するときの計算量が大きい

### 勾配降下法

- バッチ勾配降下法
    - 二乗平均誤差の変化を微分してその値が0（傾きが0）になるパラメーターを探す
        - 傾きを計算して大きかったら引き算、小さかったら足し算、をほぼ0になるまで繰り返す
    - 大量のインスタンスが存在するときの計算量が大きい
- 確率的勾配降下法
    - 二乗平均誤差ではなく、毎回ランダムに選んだ１つのインスタンスから二乗誤差を計算し、傾きの更新を0に近くまで繰り返す
    - 計算はとても早いが収束が不安定
        - 学習率を少しずつ小さくすることが有効
        - 最初はある程度大きくないと局所的な最小値にハマってしまう
- ミニバッチ勾配降下法
    - 上記の２つの手法の中間
        - ランダムに複数選んだインスタンスからなるミニバッチの二乗平均誤差で勾配降下法を用いる

### 多項式回帰

- データが単なる直線の１次式ではなく、複雑な多項式で非線形の場合
    - 線形モデルを適合させることが可能
        - scikit-learnのPolynomiaFeaturesクラスで実現
    - 例：各特徴量の二乗を新たな特徴量として追加し、元の特徴量との間の関係を見つけて回帰

### 学習曲線

- モデルを評価する際にいちいち訓練セットでの誤差と検証セットでの誤差を見て過学習や過小学習でないかを判断するのは手間がかかる
- 訓練セットの大きさと誤差の関係をグラフで表せる（学習曲線）
    - 訓練データ・検証データともに誤差が大きい→過小学習
    - 訓練データが検証データに比べて誤差が極端に小さい→過学習

### 線形モデルの正則化

- Ridge回帰
    - モデルの重さを最小限に抑える
- Lasso回帰
    - 重要度の低い特徴量の重みを０にする
- 無駄な特徴量が多い場合はLasso回帰を使えばよい

### ロジスティック回帰

- 分類に適した回帰
- 0か1かではなく、0~1の確率で値を出力
    - シグモイド関数
- 損失関数を交差エントロピーを使って計算できる（「正解ラベル×確率値の対数」のやつ！）
