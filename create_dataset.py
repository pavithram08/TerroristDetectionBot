import pandas as pd

reddit_df = pd.read_csv('./Dataset/cleanedRedditTerrorism.csv')
twitter_df = pd.read_csv('./Dataset/cleanedTwitterTerrorism.csv')
genral_df = pd.read_csv('./Dataset/cleanedRedditNonTerroism.csv')

terrorism_text = []
terrorism_label = []
for text in reddit_df['cleaned']:
    terrorism_text.append(text)
    terrorism_label.append(1)

for text in twitter_df['cleaned']:
    terrorism_text.append('text')
    terrorism_label.append(1)

non_terrorism_text = []
non_terrorism_label = []
for text in genral_df['cleaned']:
    non_terrorism_text.append(text)
    non_terrorism_label.append(0)

terrorism_df = pd.DataFrame(data={"text": terrorism_text, "label": terrorism_label})
non_terrorism_df = pd.DataFrame(data={"text": non_terrorism_text, "label": non_terrorism_label})

df = pd.concat([terrorism_df, non_terrorism_df], axis=0)
df.to_csv('./mergedData.csv', index=False)
