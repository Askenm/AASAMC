"""
The dataset is to large and the embeddings get very sparse. This script
will reduce the size of the dataset by removing the least frequent ingredients (< 20)
and only contain top 10 cuisines.

TODO: Remove too frequent ingredients.
"""
import pandas as pd

data = pd.read_json("data/raw/train.json")

# keep only ingredients that appear more than 20 times
ingredient_counts = (
    data.ingredients.apply(pd.Series).stack().value_counts()
)
ingredients_to_keep = ingredient_counts[ingredient_counts > 20].index
modified_ingredients = data[
    data.ingredients.apply(lambda x: set(x).issubset(ingredients_to_keep))
]
print(
    "#1 :Lines went from {} to {}".format(len(data), len(modified_ingredients))
)

# keep only top 10 cuisines
top_10_cuisines = modified_ingredients.cuisine.value_counts().index[:10]
modified_cuisines = modified_ingredients[modified_ingredients.cuisine.isin(top_10_cuisines)]
print("#2 : Lines went from {} to {}".format(len(modified_ingredients), len(modified_cuisines)))

# remove "salt" from ingredients
modified_cuisines.ingredients = modified_cuisines.ingredients.apply(
    lambda x: [i for i in x if i != "salt"]
)

# remove if number of ingredients is 5 or less
modified_cuisines = modified_cuisines[modified_cuisines.ingredients.apply(len) > 5]

# save the modified dataset
print("Overall lines went from {} to {}".format(len(data), len(modified_cuisines)))
modified_cuisines.to_json("data/processed/train.json", orient="records")