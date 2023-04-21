# 1章　機械学習の現状

- Gerbage in, gerbage out
    - モデルの質も大事だが訓練するデータの質はもっと大事
    - ノイズや無関係な特徴量などの見極め
- 検証セットの意義
    - モデルの評価を行うのがテストセットのみになると、そのテストセットに合わせて何度もモデルやハイパーパラメーターを調整することになり、そのテストセットだけに合ったモデルが完成する。それは精度の高いモデルとはいえない。
        - ハイパーパラメータ
            - 正則化の程度を決めるもので、訓練の前に設定する。過学習を防いでくれるが、大きすぎると鈍感な学習になる。

# 2章　エンドツーエンドの機械学習プロジェクト

- 設計の前に意識する
    - 何のための予測タスク？
        - これによってアルゴリズムの選定、評価指標、求める精度などが変わる
        - 「回帰を使って細かい値まで予測できるようにしたのに実際はカテゴリーに分ける（分類）だけでよかった」は悲しい
    - その問題に対して今はどうしてる？
        - 性能の比較対象になる
            - 性能として最低限ここは超えないといけない
- 層化抽出法
    - テストセットを作成する際にデータの中身が、重要と思われる属性の割合が現実の割合に近くなるようにする
    - 例えば男女比が2:3の組織のメンバーデータセットから何かのタスクを行うモデルを設計する場合、テストセットの男女比も2:3になるようにした方がいい、という考え方。
- 属性選びが大事
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
- データの前処理
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
- モデルの訓練と評価
    - K分割交差検証
        - 訓練セットを複数のサブセット（fold）に無作為に分割してそのうち9個で訓練を行い、1個を評価用にする。
        - ~~~python
            # 決定木モデルの評価
            from sklearn.model_selection
            score = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_aquared_error", cv=10)
            tree_rmse_scores = np.sqrt(-scores)

- モデルの微調整
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
