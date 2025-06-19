import pandas as pd

data = []
with open('data/train.ft.txt', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 1000:  # Take only first 1000 lines to keep it small
            break
        label, text = line.strip().split(' ', 1)
        sentiment = 'positive' if label == '__label__2' else 'negative'
        data.append([text, sentiment])

# Convert to DataFrame
df = pd.DataFrame(data, columns=['review', 'sentiment'])

# Save to CSV
df.to_csv('data/amazon_reviews.csv', index=False)
print("Saved to data/amazon_reviews.csv")
