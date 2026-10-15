# Exported notebook code; may contain IPython commands.
# %%
# Import necessary libraries
import sys
# Install missing packages if running in Colab or a clean environment
!{sys.executable} -m pip install -q numpy pandas scikit-learn gensim matplotlib seaborn scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
print('Setup complete. All libraries imported successfully.')


# %%
import sys
import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd

# URL of the dataset from GitHub (with fallback option)
dataset_url = "https://github.com/shenasa-ai/persian-book-recommender-dataset/blob/main/data.zip?raw=true"
hf_url = "https://huggingface.co/datasets/shenasa-ai/persian-book-recommender-dataset/resolve/main/data.zip"
zip_filename = "persian_book_dataset.zip"
extracted_folder = "extracted_data"

def download_and_extract_dataset():
    if not os.path.exists(extracted_folder):
        print("Downloading dataset from GitHub...")
        try:
            urllib.request.urlretrieve(dataset_url, zip_filename)
        except Exception as e:
            print(f"Failed to download from GitHub: {e}. Trying fallback from Hugging Face...")
            urllib.request.urlretrieve(hf_url, zip_filename)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder)
        print("Extraction complete.")
    else:
        print("Dataset already downloaded and extracted.")

# Download and extract the dataset
download_and_extract_dataset()

# Load CSVs from the extracted folder
goods_df = pd.read_csv(os.path.join(extracted_folder, 'goods.csv'))
sales_df = pd.read_csv(os.path.join(extracted_folder, 'sales.csv'))

# Basic data cleaning
goods_df['good_id'] = goods_df['good_id'].astype(int)
for col in ['brand', 'author', 'translator', 'catgory', 'tag']:
    goods_df[col] = goods_df[col].fillna('نامشخص').astype(str)

# Helper dictionary to convert digits to Persian words
# This preserves anonymized integer codes (e.g. author 115) from being removed by standard Persian text-cleaning regex
digit_map = {
    '0': 'صفر', '1': 'یک', '2': 'دو', '3': 'سه', '4': 'چهار',
    '5': 'پنج', '6': 'شش', '7': 'هفت', '8': 'هشت', '9': 'نه',
    '|': ' '
}
def num_to_persian_words(val):
    if pd.isna(val) or val == 'نامشخص':
        return 'نامشخص'
    return ''.join(digit_map.get(c, c) for c in str(val))

# Set fixed dimensions to match downstream project tasks (150 active users, 30 popular books)
num_users = 150
num_books = 30

# Get top active users and popular books from sales transactions
book_counts = sales_df['good_id'].value_counts()
user_counts = sales_df['member_id'].value_counts()

top_users = user_counts.head(num_users).index.tolist()
top_books = book_counts.head(num_books).index.tolist()

# Filter transactions for these top active users and books
filtered_sales = sales_df[sales_df['member_id'].isin(top_users) & sales_df['good_id'].isin(top_books)].copy()

# Map member/goods IDs to standard 0-indexed ranges [0, 149] and [0, 29]
user_map = {old_id: new_id for new_id, old_id in enumerate(top_users)}
book_map = {old_id: new_id for new_id, old_id in enumerate(top_books)}

filtered_sales['user_id'] = filtered_sales['member_id'].map(user_map)
filtered_sales['book_id'] = filtered_sales['good_id'].map(book_map)

# Build books metadata DataFrame
books_df = goods_df[goods_df['good_id'].isin(top_books)].copy()
books_df['book_id'] = books_df['good_id'].map(book_map)
books_df = books_df.sort_values('book_id').reset_index(drop=True)

# Create readable titles, categories (genres), and Persian descriptions
books_df['title'] = books_df['book_id'].apply(lambda x: f"کتاب شماره {x}")
books_df['genre'] = books_df['catgory'].apply(lambda x: num_to_persian_words(x.split('|')[0]))

def build_description(row):
    author_w = num_to_persian_words(row['author'])
    translator_w = num_to_persian_words(row['translator'])
    category_w = num_to_persian_words(row['catgory'])
    tag_w = num_to_persian_words(row['tag'])
    brand_w = num_to_persian_words(row['brand'])
    return f"کتاب با نویسنده {author_w} و مترجم {translator_w} و دسته بندی {category_w} و برچسب {tag_w} و برند {brand_w}"

books_df['description'] = books_df.apply(build_description, axis=1)

# Convert transaction purchase counts into ratings (scale 3 to 5)
user_book_sales = filtered_sales.groupby(['user_id', 'book_id']).size().reset_index(name='purchase_count')

def map_purchase_to_rating(count):
    if count == 1:
        return 3.0
    elif count == 2:
        return 4.0
    else:
        return 5.0

user_book_sales['rating'] = user_book_sales['purchase_count'].apply(map_purchase_to_rating)
ratings_df = user_book_sales[['user_id', 'book_id', 'rating']]

# Generate implicit actions based on transaction patterns
implicit_list = []
for idx, row in user_book_sales.iterrows():
    u, b, count = int(row['user_id']), int(row['book_id']), int(row['purchase_count'])
    implicit_list.append({'user_id': u, 'book_id': b, 'action': 'click'})
    if count >= 1:
        implicit_list.append({'user_id': u, 'book_id': b, 'action': 'read'})
    if count >= 2:
        implicit_list.append({'user_id': u, 'book_id': b, 'action': 'shelf'})

implicit_df = pd.DataFrame(implicit_list)

print(f"Loaded GitHub dataset with {len(books_df)} books.")
print(f"Explicit ratings (purchase-derived): {len(ratings_df)}")
print(f"Implicit interactions: {len(implicit_df)}")

# %%
### TODO: Task 1.1 - Complete EDA and Plotting
# 1. Calculate and print average rating
avg_rating = 0.0 # TODO

# 2. Calculate genre distribution (value_counts)
genre_counts = None # TODO

# 3. Calculate implicit feedback counts
implicit_counts = None # TODO

print(f'Average rating: {avg_rating:.2f}')
print('\nGenre distribution:')
print(genre_counts)
print('\nImplicit feedback counts:')
print(implicit_counts)

# 4. Plot Rating distribution
# TODO: Plot a histogram/bar chart of explicit ratings using matplotlib/seaborn
plt.figure(figsize=(6, 4))
# plt.bar(...) or sns.countplot(...)
plt.title('Distribution of Explicit Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


# %%
### TODO: Task 1.2 - Sparsity and Cold-Start Detection

num_users = 150
num_books = 30

# 1. Build the user-item interaction matrix R (size: num_users x num_books)
# Fill with 0 for missing values
R = np.zeros((num_users, num_books))
for idx, row in ratings_df.iterrows():
    # TODO: Populate R with ratings
    pass

# 2. Calculate Sparsity
sparsity = 0.0 # TODO
print(f'User-Item matrix sparsity: {sparsity:.2%} ({sparsity:.4f})')

# 3. Find Cold-Start Users and Items (rated <= 1 times)
cold_users = [] # TODO
cold_books = [] # TODO

print(f'Number of cold-start users: {len(cold_users)}')
print(f'Cold-start user IDs: {cold_users}')
print(f'Number of cold-start books: {len(cold_books)}')
print(f'Cold-start book IDs: {cold_books}')


# %%
def clean_persian_text(text):
    # TODO: Remove special characters, numbers, punctuations, and return cleaned space-separated string
    # Hint: You can use regex to keep only Persian characters and spaces: r'[^آ-ی\s]'
    cleaned = text # TODO
    return cleaned

# Clean descriptions
cleaned_descriptions = books_df['description'].apply(clean_persian_text).tolist()

### TODO: Task 2.1 - Compute TF-IDF Features
# Initialize TfidfVectorizer with ngram_range=(1,2)
tfidf_vec = TfidfVectorizer() # TODO (add parameters)
tfidf_matrix = None # TODO: fit_transform

print(f'TF-IDF Feature Matrix Shape: {tfidf_matrix.shape}')


# %%
### TODO: Task 2.2 - Word Embeddings Feature Matrix
tokenized_descriptions = [desc.split() for desc in cleaned_descriptions]

# 1. Train Word2Vec model
w2v_model = None # TODO

# 2. Calculate average word vectors for each book
embedding_matrix = np.zeros((num_books, 32))
for i, tokens in enumerate(tokenized_descriptions):
    # TODO: Average the word vectors of tokens that exist in w2v_model.wv
    pass

print(f'Word2Vec Embedding Matrix Shape: {embedding_matrix.shape}')


# %%
### TODO: Task 2.3 - Content-Based Recommender Function
def recommend_content_based(user_id, top_k=5, feature_matrix=tfidf_matrix.toarray()):
    # 1. Get positive books and their user ratings
    # Filter ratings_df for this user
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    
    # Return empty list or fallback if user has no ratings (Cold Start)
    if len(user_ratings) == 0:
        # Return top_k popular books as fallback
        popular_books = ratings_df.groupby('book_id').size().sort_values(ascending=False).index.tolist()
        return [b for b in popular_books if b < num_books][:top_k]
        
    # 2. Build User Profile
    # user_profile = weighted sum of feature_matrix[book_id] * weight
    # weight = rating - 2.5
    user_profile = np.zeros(feature_matrix.shape[1])
    
    # TODO: Implement user profile construction
    
    # 3. Calculate similarity of User Profile to all books
    # similarities = cosine_similarity(user_profile, feature_matrix)
    similarities = np.zeros(num_books) # TODO
    
    # 4. Filter out already rated books by the user and sort
    # TODO
    
    recommended_ids = [] # TODO: top_k recommendations
    return recommended_ids

# Test for User ID 10
test_user = 10
rec_books = recommend_content_based(test_user, top_k=3)
print(f'Recommendations for User {test_user}: {rec_books}')
print('Recommended titles:', books_df.loc[rec_books, 'title'].tolist())


# %%
### TODO: Task 3.1 - User-Based & Item-Based CF

# 1. Calculate Pearson Correlation similarity matrix between users
def compute_user_similarities(R):
    # Note: R is (num_users, num_books)
    # Center R by subtracting mean rating of each user (only for rated items, ignore 0s)
    R_centered = np.zeros_like(R)
    for u in range(R.shape[0]):
        rated_indices = R[u] > 0
        if rated_indices.any():
            R_centered[u, rated_indices] = R[u, rated_indices] - np.mean(R[u, rated_indices])
            
    # Compute cosine similarity on centered ratings to get Pearson correlation
    # TODO: user_sim of shape (num_users, num_users)
    user_sim = np.zeros((R.shape[0], R.shape[0]))
    return user_sim

user_sim = compute_user_similarities(R)

# 2. Calculate Jaccard similarity matrix between books
def compute_jaccard_item_similarities(R_bin):
    # R_bin is a binary matrix (shape: num_users, num_books) showing positive interaction (ratings >= 3.0)
    # TODO: item_sim of shape (num_books, num_books)
    item_sim = np.zeros((R_bin.shape[1], R_bin.shape[1]))
    return item_sim

R_bin = (R >= 3.0).astype(float)
item_sim = compute_jaccard_item_similarities(R_bin)

# 3. Rating prediction formula for User-Based CF:
# R_pred(u, i) = mean_R(u) + sum(sim(u, v) * R_centered(v, i)) / sum(|sim(u, v)|)
def predict_rating_user_based(user_id, book_id, R, user_sim, k_neighbors=10):
    # TODO: Predict explicit rating
    pred_rating = 0.0
    return pred_rating

print('User Sim shape:', user_sim.shape)
print('Item Sim shape:', item_sim.shape)


# %%
class MatrixFactorizationSGD:
    def __init__(self, num_users, num_items, num_factors=8, lr=0.01, reg=0.05, epochs=30):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        
        # Initialize Latent Matrices randomly
        self.P = np.random.normal(scale=1./self.num_factors, size=(self.num_users, self.num_factors))
        self.Q = np.random.normal(scale=1./self.num_factors, size=(self.num_items, self.num_factors))
        
    def fit(self, train_ratings_df):
        # train_ratings_df contains columns: user_id, book_id, rating
        for epoch in range(self.epochs):
            # Shuffle the data each epoch
            shuffled_df = train_ratings_df.sample(frac=1.0).reset_index(drop=True)
            for idx, row in shuffled_df.iterrows():
                u = int(row['user_id'])
                i = int(row['book_id'])
                r = row['rating']
                
                # TODO: Implement SGD update rules
                # 1. Compute prediction error: error = r - P_u^T * Q_i
                # 2. Update P_u
                # 3. Update Q_i
                pass
                
    def predict(self, user_id, item_id):
        # TODO: Return dot product P_u^T * Q_i + clamp it between 1.0 and 5.0
        pred = 3.0
        return pred

# Instantiate and train MF-SGD
mf = MatrixFactorizationSGD(num_users, num_books, num_factors=10, epochs=25)
mf.fit(ratings_df)
print('Matrix Factorization SGD trained successfully.')


# %%
class BPRMF:
    def __init__(self, num_users, num_items, num_factors=8, lr=0.05, reg=0.01, epochs=15):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        
        # Initialize factors
        self.W = np.random.normal(scale=1./self.num_factors, size=(self.num_users, self.num_factors)) # user factors
        self.H = np.random.normal(scale=1./self.num_factors, size=(self.num_items, self.num_factors)) # item factors
        
    def fit(self, implicit_df):
        # 1. Prepare positive interactions dictionary for fast lookup
        # Positive actions are read or shelf
        pos_df = implicit_df[implicit_df['action'].isin(['read', 'shelf'])]
        user_pos = pos_df.groupby('user_id')['book_id'].apply(set).to_dict()
        
        # Remove users with no positive interactions to avoid errors
        users_with_pos = list(user_pos.keys())
        if len(users_with_pos) == 0:
            return
            
        # 2. Main SGD training loop
        num_triplets = len(pos_df) * 3  # number of triplets to sample per epoch
        
        for epoch in range(self.epochs):
            for _ in range(num_triplets):
                # Sample random user u with positive interactions
                u = np.random.choice(users_with_pos)
                pos_items = user_pos[u]
                if len(pos_items) == 0: continue
                
                # Sample positive item i
                i = np.random.choice(list(pos_items))
                
                # Sample negative item j (not in user's positive interactions)
                j = np.random.choice(self.num_items)
                while j in pos_items:
                    j = np.random.choice(self.num_items)
                    
                # TODO: Implement BPR updates
                # 1. Compute prediction score: x_uij = W_u^T * H_i - W_u^T * H_j
                # 2. Compute sigmoid derivative helper: sigmoid_deriv = 1 / (1 + exp(x_uij))
                # 3. Apply gradient updates to W_u, H_i, H_j
                pass
                
    def recommend(self, user_id, top_k=5):
        # TODO: Compute ranking scores for all items: score = W_u^T * H
        # Return top_k item indices sorted descending
        scores = np.dot(self.W[user_id], self.H.T)
        return np.argsort(scores)[::-1][:top_k].tolist()

# Instantiate and train BPR
bpr = BPRMF(num_users, num_books, num_factors=8, epochs=20)
bpr.fit(implicit_df)
print('BPR recommender trained successfully.')


# %%
class HybridRecommender:
    def __init__(self, cb_func, mf_model, alpha=0.5):
        self.cb_func = cb_func # recommend_content_based function
        self.mf = mf_model     # trained MatrixFactorizationSGD instance
        self.alpha = alpha     # weight factor for Weighted Hybrid
        
    def recommend(self, user_id, top_k=5):
        # TODO: Implement your choice of Hybrid Model (e.g. Switching or Weighted Hybrid)
        # Let's write a switching hybrid based on user profile activity:
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        
        # If user has <= 1 ratings, we switch to Content-Based
        if len(user_ratings) <= 1:
            # Content-based recommendations
            return self.cb_func(user_id, top_k=top_k)
        else:
            # CF-based recommendation: predict ratings for all items not yet rated
            # and sort by predicted rating
            rated_items = set(user_ratings['book_id'].tolist())
            predictions = []
            for item_id in range(num_books):
                if item_id not in rated_items:
                    pred = self.mf.predict(user_id, item_id)
                    predictions.append((item_id, pred))
            
            # Sort descending by predicted rating
            predictions.sort(key=lambda x: x[1], reverse=True)
            return [item_id for item_id, _ in predictions[:top_k]]

# Instantiate Hybrid Recommender
hybrid = HybridRecommender(recommend_content_based, mf, alpha=0.4)
rec_hybrid = hybrid.recommend(10, top_k=3)
print('Hybrid recommendations for User 10:', rec_hybrid)


# %%
# Split ratings into Train (80%) and Test (20%)
from sklearn.model_selection import train_test_split
train_ratings, test_ratings = train_test_split(ratings_df, test_size=0.2, random_state=42)
print(f'Train ratings shape: {train_ratings.shape}')
print(f'Test ratings shape: {test_ratings.shape}')


# %%
### TODO: Task 5.1 - RMSE and MAE evaluation
def evaluate_rating_prediction(model, test_df):
    # model is a trained model with a .predict(user_id, item_id) function
    # test_df contains columns user_id, book_id, rating
    
    errors = []
    abs_errors = []
    
    for idx, row in test_df.iterrows():
        u = int(row['user_id'])
        i = int(row['book_id'])
        r_true = row['rating']
        
        r_pred = model.predict(u, i) # TODO: get model prediction
        
        # TODO: Calculate errors
        errors.append(0.0)
        abs_errors.append(0.0)
        
    rmse = 0.0 # TODO
    mae = 0.0 # TODO
    return rmse, mae

# Train MF on train_ratings only
mf_eval = MatrixFactorizationSGD(num_users, num_books, num_factors=10, epochs=25)
mf_eval.fit(train_ratings)

mf_rmse, mf_mae = evaluate_rating_prediction(mf_eval, test_ratings)
print(f'Matrix Factorization SGD -> RMSE: {mf_rmse:.4f}, MAE: {mf_mae:.4f}')


# %%
### TODO: Task 5.2 - Ranking Metrics Evaluation
def evaluate_ranking(recommender_model, test_ratings_df, top_k=5):
    # recommender_model has a .recommend(user_id, top_k) function
    precisions = []
    recalls = []
    aps = []
    ndcgs = []
    hits = []
    
    # Get positive ground truth books for each user in the test set (rating >= 4)
    test_pos = test_ratings_df[test_ratings_df['rating'] >= 4.0]
    user_test_pos = test_pos.groupby('user_id')['book_id'].apply(set).to_dict()
    
    for u in range(num_users):
        if u not in user_test_pos or len(user_test_pos[u]) == 0:
            continue
            
        rel = user_test_pos[u]
        rec = recommender_model.recommend(u, top_k=top_k)
        
        # TODO: Calculate Metrics
        # 1. Precision@K
        # 2. Recall@K
        # 3. Average Precision (AP)
        # 4. NDCG@K
        # 5. Hit Rate (1 if hit, else 0)
        
        precision_k = 0.0 # TODO
        recall_k = 0.0 # TODO
        ap = 0.0 # TODO
        ndcg = 0.0 # TODO
        hit = 0 # TODO
        
        precisions.append(precision_k)
        recalls.append(recall_k)
        aps.append(ap)
        ndcgs.append(ndcg)
        hits.append(hit)
        
    return {
        'Precision@K': np.mean(precisions),
        'Recall@K': np.mean(recalls),
        'MAP': np.mean(aps),
        'nDCG': np.mean(ndcgs),
        'HitRate': np.mean(hits)
    }

# Wrap content-based recommender to expose a .recommend(user_id, top_k) signature
class CBF_Wrapper:
    def recommend(self, user_id, top_k):
        return recommend_content_based(user_id, top_k)

cbf_model = CBF_Wrapper()

# Compare Content-Based vs Hybrid on test set
cbf_metrics = evaluate_ranking(cbf_model, test_ratings, top_k=5)
hybrid_metrics = evaluate_ranking(hybrid, test_ratings, top_k=5)
bpr_metrics = evaluate_ranking(bpr, test_ratings, top_k=5)

print('Ranking Metrics (Top 5):')
print('Content-Based CF:', cbf_metrics)
print('Hybrid CF Model:', hybrid_metrics)
print('BPR Model (Implicit):', bpr_metrics)


# %%
### TODO: Task 5.3 - Comparison Plotting
# 1. Create a summary DataFrame
results_data = {
    'Model': ['Content-Based', 'BPR (Implicit)', 'Hybrid (Switching)'],
    'Precision@5': [cbf_metrics['Precision@K'], bpr_metrics['Precision@K'], hybrid_metrics['Precision@K']],
    'Recall@5': [cbf_metrics['Recall@K'], bpr_metrics['Recall@K'], hybrid_metrics['Recall@K']],
    'nDCG@5': [cbf_metrics['nDCG'], bpr_metrics['nDCG'], hybrid_metrics['nDCG']],
    'HitRate@5': [cbf_metrics['HitRate'], bpr_metrics['HitRate'], hybrid_metrics['HitRate']]
}
df_results = pd.DataFrame(results_data)
print(df_results)

# 2. Plot comparison bar chart of metrics using seaborn/matplotlib
# TODO: Plot Precision@5 and Recall@5 for each model
plt.figure(figsize=(10, 5))
# df_melted = pd.melt(df_results, id_vars='Model', value_vars=['Precision@5', 'Recall@5'])
# sns.barplot(x='Model', y='value', hue='variable', data=df_melted)
plt.title('Recommender Systems Performance Comparison')
plt.ylabel('Score')
plt.show()

