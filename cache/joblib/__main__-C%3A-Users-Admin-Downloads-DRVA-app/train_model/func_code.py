# first line: 34
@memory.cache
def train_model():
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, min_df=2)
    X = vectorizer.fit_transform(df["review"])
    y = LabelEncoder().fit_transform(df["condition"])
    model = lgb.train({"objective": "multiclass", "num_class": len(np.unique(y)), "metric": "multi_logloss", "verbose": -1}, lgb.Dataset(X, label=y))
    return model, vectorizer
