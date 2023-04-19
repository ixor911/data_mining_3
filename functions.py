import matplotlib.pyplot as plt
import pandas as pd


def get_colors_result(result: list) -> list:
    colors = ['blue', 'red', 'green', 'orange', 'brown', 'purple']
    color_results = []
    for item in result:
        color_results.append(colors[item])

    return color_results


def get_results(models: list) -> list:
    results = []
    for model in models:
        results.append(model.get_result())

    return results


def normal_results(models: list) -> dict:
    result = {}
    for model in models:
        result[model.name] = model.get_result()

    return result


def show_results(data: pd.DataFrame, results: dict):
    x = list(data['x'])
    y = list(data['y'])

    index = 0
    for key in results.keys():
        index += 1
        colors = get_colors_result(results.get(key))

        plt.subplot(2, 3, index)
        plt.scatter(x, y, c=colors)
        plt.gca().set_title(key)

    plt.show()











