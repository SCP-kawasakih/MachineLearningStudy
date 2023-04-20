# 1章　機械学習の現状

- Gerbage in, gerbage out
    - モデルの質も大事だが訓練するデータの質はもっと大事
    - ノイズや無関係な特徴量などの見極め
- 検証セットの意義
    - モデルの評価を行うのがテストセットのみになると、そのテストセットに合わせて何度もモデルやハイパーパラメーターを調整することになり、そのテストセットだけに合ったモデルが完成する。それは精度の高いモデルとはいえない。

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
    - 例：「人工」と「世帯数」から「１世帯当たりの人数」など
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
