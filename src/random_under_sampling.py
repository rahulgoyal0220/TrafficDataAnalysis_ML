red = len(df[df['Congestion_Status'] == 1])
green = len(df[df['Congestion_Status'] == 0])
yellow = len(df[df['Congestion_Status'] == 2])

green_indices = df[df.Congestion_Status == 0].index
yellow_indices = df[df.Congestion_Status == 2].index
random_indices1 = np.random.choice(yellow_indices,red, replace=False)
random_indices2 = np.random.choice(green_indices,red, replace=False)
red_indices = df[df.Congestion_Status == 1].index
under_sample_indices = np.concatenate([red_indices,random_indices1,random_indices2])
under_sample = df.loc[under_sample_indices]
