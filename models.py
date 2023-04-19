from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
import pandas as pd


class Kmeans:
    def __init__(self, data: pd.DataFrame, clusters: int):
        self.model = KMeans(n_init="auto", n_clusters=clusters)
        self.model.fit(data)
        self.result = list(self.model.predict(data))
        self.name = "k-means"

    def get_result(self) -> list:
        return self.result

    def predict(self, df: pd.DataFrame) -> list:
        return list(self.model.predict(df))


class Kmedoids:
    def __init__(self, data: pd.DataFrame, clusters: int):
        self.model = KMedoids(n_clusters=clusters)
        self.model.fit(data)
        self.result = list(self.model.predict(data))
        self.name = "k-medians"

    def get_result(self) -> list:
        return self.result

    def predict(self, df: pd.DataFrame) -> list:
        return list(self.model.predict(df))


class Link:
    def __init__(self, data: pd.DataFrame, clusters: int, linkage: str):
        self.model = AgglomerativeClustering(n_clusters=clusters, linkage=linkage)
        self.result = list(self.model.fit_predict(data))
        self.name = f"{linkage}-link"

    def get_result(self) -> list:
        return self.result

    def predict(self, df: pd.DataFrame) -> list:
        return list(self.model.fit_predict(df))











