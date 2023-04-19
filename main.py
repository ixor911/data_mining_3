import json
import models
import functions
import pandas as pd


data = json.load(open("data.json"))

for variant in data.keys():
    variant_df = pd.DataFrame(data=data.get(variant))

    models_list = [
        models.Link(variant_df, 2, 'single'),
        models.Link(variant_df, 2, 'complete'),
        models.Link(variant_df, 2, 'average'),
        models.Kmeans(variant_df, 3),
        models.Kmedoids(variant_df, 3)
    ]

    normal_results = functions.normal_results(models_list)

    print(f"{variant}:\n"
          f"\tX: {list(variant_df['x'])}\n"
          f"\tY: {list(variant_df['y'])}\n")
    for model in normal_results:
        print(f"\t{model}: {normal_results.get(model)}")

    functions.show_results(variant_df, normal_results)
    print("===========================================================")
