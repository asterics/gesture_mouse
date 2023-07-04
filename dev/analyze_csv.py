import pandas as pd
import matplotlib.pyplot as plt

directions = ["test"]
csv_base = "../tests/Patrick Link"
pose = "pnp"
length=300

ear_names = list(map(lambda i: f"ear_{i}", range(17)))
corrected_ear_names = list(map(lambda i: f"corrected_ear_{i}", range(17)))
combined_df = pd.DataFrame()
for direction in directions:
    df = pd.read_csv(f"{csv_base}/{pose}_{direction}.csv", delimiter=";")
    print(len(df))
    start = (len(df)-length)//2
    df = df.iloc[start:start+length]
    df["direction"]=direction
    combined_df = pd.concat([combined_df,df])
    ear_values = df[ear_names]
    corrected_ear_values = df[corrected_ear_names]
    #print(f"{direction}: {ear_values.mean()}")
    #print(corrected_ear_values.describe())
combined_df["direction"] =combined_df["direction"].astype("category")
normalization = combined_df[combined_df["direction"]=="neutral"].mean(numeric_only=True)
#combined_df[ear_names] = combined_df[ear_names]/normalization[ear_names]
#combined_df[corrected_ear_names] = combined_df[corrected_ear_names]/normalization[corrected_ear_names]
#combined_df["direction"] = combined_df["direction"].astype("category", ordered=True)
grouped = combined_df.groupby("direction")

tracking_stats = grouped[ear_names].describe()
print(tracking_stats)
tracking_stats.to_csv("tracking_summery.csv")


# fig, ax = plt.subplots(nrows=6,ncols=3, figsize=(15,15))

# for i, name in enumerate(zip(ear_names,corrected_ear_names)):
#     combined_df.boxplot(column=name, by="direction", ax=ax[i//3,i%3])


# fig.tight_layout()
# plt.show()

fig, ax = plt.subplots(nrows=6,ncols=3, figsize=(15,15))

for i, name in enumerate(zip(ear_names, corrected_ear_names)):
    combined_df.boxplot(column=name[0], by="direction", ax=ax[i//3,i%3])
    combined_df.boxplot(column=name[1], by="direction", ax=ax[i//3,i%3])

fig.tight_layout()
plt.show()
for name, corrected_name in zip(ear_names,corrected_ear_names):
    print(combined_df[[name,corrected_name]].describe())

for ear, corrected_ear in zip(ear_names,corrected_ear_names):
    grouped.plot.scatter(x=ear,y=corrected_ear)
    plt.show()



