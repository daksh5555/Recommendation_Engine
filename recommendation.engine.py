import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Check and configure TensorFlow to use the GPU to enhace the processing
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs detected:")
    for gpu in gpus:
        print(f"  - {gpu.name}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs detected.")

# Load the dataset
df = pd.read_csv('socialverse_data_output.csv')

# Handle missing 'Title' column
df['Title_text'] = df['Title'].apply(lambda x: str(x) if isinstance(x, str) else 'NO_TITLE')
df['Title_number'] = df['Title'].apply(lambda x: x if isinstance(x, (int, float)) else np.nan)

# Impute missing values in numerical columns
for col in ['View Count', 'Share Count', 'Exit Count', 'Average Rating']:
    df[col] = df[col].replace(0, np.nan).fillna(df[col].mean())

# Drop the original 'Title' column
df = df.drop('Title', axis=1)

# Initialize TF-IDF vectorizer for content-based filtering
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=100)

# Process text data with TF-IDF
text_features = tfidf_vectorizer.fit_transform(df['Title_text']).toarray()
text_features_df = pd.DataFrame(text_features, columns=tfidf_vectorizer.get_feature_names_out()).fillna(0)

# Combine numerical features with text features
numerical_features = df[['View Count', 'Share Count', 'Exit Count', 'Average Rating', 'Title_number']]
features_df = pd.concat([numerical_features, text_features_df], axis=1)

# Normalize numerical features
scaler = StandardScaler()
features_df[['View Count', 'Share Count', 'Exit Count', 'Average Rating', 'Title_number']] = scaler.fit_transform(
    features_df[['View Count', 'Share Count', 'Exit Count', 'Average Rating', 'Title_number']]
)

# Save preprocessed features to CSV
features_df.to_csv('preprocessed_features.csv', index=False)

# Prepare target variable
y = df['Average Rating']

# Split data into train and test sets for collaborative filtering
X_train_collab, X_test_collab, y_train_collab, y_test_collab = train_test_split(
    features_df.reset_index(drop=True), y, test_size=0.2, random_state=42
)

# Define neural network model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(features_df.shape[1],)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Implement early stopping and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

# Cross-validation for hyperparameter tuning
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train_collab), 1):
    X_train_fold, X_val_fold = X_train_collab.iloc[train_index], X_train_collab.iloc[val_index]
    y_train_fold, y_val_fold = y_train_collab.iloc[train_index], y_train_collab.iloc[val_index]
    
    # Create a new model for each fold
    model = create_model()
    
  
    
    # Train the model for each fold
    history = model.fit(
        X_train_fold, y_train_fold,
        epochs=50,  # Number of epochs for each fold
        validation_data=(X_val_fold, y_val_fold),
        batch_size=32,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1  # Show epoch-by-epoch progress
    )
    
    # Evaluate the model on validation data
    val_loss = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    fold_results.append(val_loss)
    
# Print average validation loss across all folds
average_val_loss = np.mean(fold_results)
print(f'Average validation loss across all folds: {average_val_loss:.4f}')

# Define normalization functions
def normalize_mse(mse, y_train):
    y_max = y_train.max()
    y_min = y_train.min()
    range_y = y_max - y_min
    if range_y == 0:
        return 0
    return mse / range_y

def normalize_mse_fixed(mse, max_possible_error):
    return mse / max_possible_error

def normalize_collaborative_score(score, max_score=5):
    return min(1, max(0, score / max_score))

def normalize_content_based_score(score):
    return (score + 1) / 2

# Function to compute content-based similarity recommendations
def content_based_recommendation(user):
    user_data = df[df['Username'] == user]
    
    if user_data.empty:
        return pd.DataFrame(columns=['ID', 'Similarity Score'])
    
    user_video_ids = user_data['ID'].values
    user_indices = user_data.index

    # Compute cosine similarity between user video features and all videos in the dataset
    user_features = features_df.loc[user_indices].fillna(0)
    similarities = cosine_similarity(user_features, features_df.fillna(0))
    
    recommendations = []
    
    for i, user_index in enumerate(user_indices):
        similar_videos = similarities[i]
        
        # Create DataFrame for this user's similar videos
        similar_videos_df = pd.DataFrame({
            'ID': df['ID'],
            'Similarity Score': similar_videos
        }).sort_values(by='Similarity Score', ascending=False)
        
        # Filter out videos the user has already watched
        similar_videos_df = similar_videos_df[~similar_videos_df['ID'].isin(user_video_ids)]
        
        recommendations.append(similar_videos_df)
    
    # Concatenate all recommendations into a single DataFrame
    recommendations = pd.concat(recommendations)
    
    # Drop duplicates and sort by similarity score
    recommendations = recommendations.drop_duplicates(subset=['ID']).sort_values(by='Similarity Score', ascending=False)
    
    return recommendations


def get_default_recommendations(top_n=50):
    # Example: Return top N popular items based on view count
    popular_items = df.groupby('ID').agg({'View Count': 'sum'}).reset_index()
    popular_items = popular_items.sort_values(by='View Count', ascending=False).head(top_n)
    popular_items = popular_items.rename(columns={'View Count': 'Final Score'})
    popular_items['Recommendation Type'] = 'Default'
    return popular_items


# Function to get hybrid recommendations for a given user
# Function to get hybrid recommendations for a given user
def get_hybrid_recommendations(user, rating_threshold=4.0, weight_collaborative=0.5, top_n=50):
    # Check if the user exists in the dataset
    if user not in df['Username'].values:
        # Handle new user scenario
        return get_default_recommendations(top_n)
    
    # Get collaborative filtering recommendations
    user_data = df[df['Username'] == user]
    if user_data.empty:
        return f"User {user} not found"
    
    # Get the intersection of user_data.index and X_train_collab.index
    common_index = user_data.index.intersection(X_train_collab.index)

    # Filter user_data to only include common index values
    user_data = user_data.loc[common_index]

    # Now you can safely access X_train_collab using user_data.index
    user_features = X_train_collab.loc[user_data.index]
    
    predicted_ratings = model.predict(user_features)
    collaborative_recommendations = pd.DataFrame({
        'ID': user_data['ID'],
        'Predicted Rating': predicted_ratings.flatten()
    })
    
    # Normalize collaborative scores
    collaborative_recommendations['Predicted Rating'] = collaborative_recommendations['Predicted Rating'].apply(
        lambda x: normalize_collaborative_score(x, max_score=5)
    )
    
    # Filter out low-rating predictions
    collaborative_recommendations = collaborative_recommendations[
        collaborative_recommendations['Predicted Rating'] >= rating_threshold
    ]
    
    # Get all content-based recommendations
    content_based_recommendations = content_based_recommendation(user)
    
    # Prepare combined recommendations
    combined_recommendations = {}
    
    # Get all video IDs from both recommendations
    all_video_ids = set(collaborative_recommendations['ID']).union(set(content_based_recommendations['ID']))
    
    for video_id in all_video_ids:
        collaborative_row = collaborative_recommendations[collaborative_recommendations['ID'] == video_id]
        content_based_row = content_based_recommendations[content_based_recommendations['ID'] == video_id]
        
        if not collaborative_row.empty and not content_based_row.empty:
            # Both scores are available, combine them
            collaborative_score = collaborative_row['Predicted Rating'].values[0]
            content_based_score = normalize_content_based_score(content_based_row['Similarity Score'].values[0])
            combined_score = (collaborative_score * weight_collaborative) + (content_based_score * (1 - weight_collaborative))
            recommendation_type = 'Combined'
        elif not collaborative_row.empty:
            # Only collaborative score is available
            combined_score = collaborative_row['Predicted Rating'].values[0]
            recommendation_type = 'Collaborative'
        elif not content_based_row.empty:
            # Only content-based score is available
            content_based_score = normalize_content_based_score(content_based_row['Similarity Score'].values[0])
            combined_score = content_based_score
            recommendation_type = 'Content-Based'
        else:
            # No score available for this video
            continue
        
        combined_recommendations[video_id] = {'score': combined_score, 'type': recommendation_type}
    
    # Normalize combined scores to [0, 1]
    if combined_recommendations:
        max_combined_score = max(score['score'] for score in combined_recommendations.values())
        min_combined_score = min(score['score'] for score in combined_recommendations.values())
        
        for video_id, details in combined_recommendations.items():
            if max_combined_score != min_combined_score:
                normalized_score = (details['score'] - min_combined_score) / (max_combined_score - min_combined_score)
            else:
                normalized_score = details['score']
            combined_recommendations[video_id]['score'] = normalized_score
    
    # Finalize recommendations: use combined if score is 1, otherwise use the highest individual score
    final_recommendations = {}
    
    for video_id, details in combined_recommendations.items():
        if details['score'] == 1:
            final_score = details['score']
            final_type = 'Combined'
        else:
            # Get individual scores
            collaborative_row = collaborative_recommendations[collaborative_recommendations['ID'] == video_id]
            content_based_row = content_based_recommendations[content_based_recommendations['ID'] == video_id]
            
            if not collaborative_row.empty and not content_based_row.empty:
                collaborative_score = collaborative_row['Predicted Rating'].values[0]
                content_based_score = normalize_content_based_score(content_based_row['Similarity Score'].values[0])
                
                if collaborative_score > content_based_score:
                    final_score = collaborative_score
                    final_type = 'Collaborative'
                else:
                    final_score = content_based_score
                    final_type = 'Content-Based'
            elif not collaborative_row.empty:
                final_score = collaborative_row['Predicted Rating'].values[0]
                final_type = 'Collaborative'
            elif not content_based_row.empty:
                final_score = normalize_content_based_score(content_based_row['Similarity Score'].values[0])
                final_type = 'Content-Based'
            else:
                continue
        
        final_recommendations[video_id] = {'score': final_score, 'type': final_type}
    
    # Convert final recommendations into a DataFrame
    final_recommendations_df = pd.DataFrame.from_dict(final_recommendations, orient='index').reset_index()
    final_recommendations_df.columns = ['Video ID', 'Final Score', 'Recommendation Type']
    final_recommendations_df = final_recommendations_df.sort_values(by='Final Score', ascending=False).head(top_n)
    
    return final_recommendations_df


def get_all_users_recommendations(rating_threshold=4.0, weight_collaborative=0.5, top_n=50):
    all_recommendations = {}
    unique_users = df['Username'].unique()
    
    # Get default recommendations for new users
    default_recommendations = get_default_recommendations(top_n)
    
    for user in unique_users:
        recommendations = get_hybrid_recommendations(user, rating_threshold, weight_collaborative, top_n)
        all_recommendations[user] = recommendations
    
    # Handle new users not in the dataset
    all_existing_users = set(unique_users)
    
    # For demonstration, assume we know which users are new (this could be dynamic based on user interaction)
    new_users = set(df['Username'].unique()) - all_existing_users
    
    for new_user in new_users:
        all_recommendations[new_user] = default_recommendations
    
    return all_recommendations

# Generate recommendations for all users including new ones
all_users_recommendations = get_all_users_recommendations()

# Print recommendations for each user
for user, recommendations in all_users_recommendations.items():
    print(f"Recommendations for user {user}:\n", recommendations, "\n")

# Predict ratings for the test set (collaborative filtering)
y_pred = model.predict(X_test_collab)

# Calculate MSE, MAE, and RMSE
mse = mean_squared_error(y_test_collab, y_pred)
mae = mean_absolute_error(y_test_collab, y_pred)
rmse = np.sqrt(mse)

# Normalize MSE using range of target values
normalized_mse = normalize_mse(mse, y_train_collab)
# Print the evaluation metrics
print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Normalized MSE: {normalized_mse:.4f}")

# Evaluate Precision and Recall for each user
def precision_at_k(recommendations, user_data, k=10):
    relevant_items = set(user_data['ID'])
    top_k_recommendations = recommendations.head(k)
    recommended_items = set(top_k_recommendations['Video ID'])
    
    true_positives = len(relevant_items.intersection(recommended_items))
    precision = true_positives / k if k > 0 else 0
    
    return precision

def recall_at_k(recommendations, user_data, k=10):
    relevant_items = set(user_data['ID'])
    top_k_recommendations = recommendations.head(k)
    recommended_items = set(top_k_recommendations['Video ID'])
    
    true_positives = len(relevant_items.intersection(recommended_items))
    recall = true_positives / len(relevant_items) if len(relevant_items) > 0 else 0
    
    return recall

# Calculate Precision and Recall for each user
def evaluate_recommendations(user, top_n=50):
    recommendations = get_hybrid_recommendations(user, top_n=top_n)
    user_data = df[df['Username'] == user]
    
    precision = precision_at_k(recommendations, user_data, k=10)
    recall = recall_at_k(recommendations, user_data, k=10)
    
    return precision, recall

# Print Precision and Recall for each user
for user in df['Username'].unique():
    precision, recall = evaluate_recommendations(user)
    print(f"User {user}: Precision@10 = {precision:.2f}, Recall@10 = {recall:.2f}")

# if train model based on best learning rate

# step 1: Define a Function to Train the Model with Different Learning Rates
'''
# if train model based on best learning rate
import matplotlib.pyplot as plt
import numpy as np

# Function to train the model and return the loss for different learning rates
def find_best_learning_rate(X_train, y_train, X_val, y_val, min_lr=1e-5, max_lr=1e-1, steps=100):
    lr_list = np.logspace(np.log10(min_lr), np.log10(max_lr), num=steps)
    loss_list = []

    for lr in lr_list:
        model = create_model()  # Create a new model instance
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')
        
        history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_val, y_val), verbose=0)
        val_loss = history.history['val_loss'][-1]  # Get the last validation loss
        loss_list.append(val_loss)
    
    return lr_list, loss_list

# Run the function to find the best learning rate
lr_list, loss_list = find_best_learning_rate(X_train_collab, y_train_collab, X_test_collab, y_test_collab)

# Plotting the loss vs learning rate
plt.figure(figsize=(10, 6))
plt.plot(lr_list, loss_list, marker='o')
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Loss')
plt.title('Learning Rate vs Validation Loss')
plt.show()'''

# step2:  Train the Final Model with the Best Learning Rate
'''
best_lr = lr_list[np.argmin(loss_list)]  # Get the learning rate with the lowest validation loss

# Train the final model using the best learning rate
model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_lr), loss='mean_squared_error')

history = model.fit(X_train_collab, y_train_collab, epochs=50, validation_data=(X_test_collab, y_test_collab),
                    batch_size=32, callbacks=[early_stopping, lr_scheduler])

'''
# if want more advance hybrid and accurate you can use this replacing above
'''
def get_hybrid_recommendations(user, rating_threshold=4.0, weight_collaborative=0.5, top_n=50):
    
    # Get collaborative filtering recommendations

    user_data = df[df['Username'] == user]
    if user_data.empty:
        return f"User {user} not found"
    
    # Get the intersection of user_data.index and X_train_collab.index

    common_index = user_data.index.intersection(X_train_collab.index)

    # Filter user_data to only include common index values

    user_data = user_data.loc[common_index]

    # Now you can safely access X_train_collab using user_data.index

    user_features = X_train_collab.loc[user_data.index]
    
    predicted_ratings = model.predict(user_features)
    collaborative_recommendations = pd.DataFrame({
        'ID': user_data['ID'],
        'Predicted Rating': predicted_ratings.flatten()
    })
    
    # Normalize collaborative scores

    collaborative_recommendations['Predicted Rating'] = collaborative_recommendations['Predicted Rating'].apply(
        lambda x: normalize_collaborative_score(x, max_score=5)
    )
    
    # Filter out low-rating predictions

    collaborative_recommendations = collaborative_recommendations[
        collaborative_recommendations['Predicted Rating'] >= rating_threshold
    ]
    
    # Get all content-based recommendations

    content_based_recommendations = content_based_recommendation(user)
    
    # Prepare combined recommendations

    combined_recommendations = {}
    
    # Get all video IDs from both recommendations
    all_video_ids = set(collaborative_recommendations['ID']).union(set(content_based_recommendations['ID']))
    
    for video_id in all_video_ids:
        collaborative_row = collaborative_recommendations[collaborative_recommendations['ID'] == video_id]
        content_based_row = content_based_recommendations[content_based_recommendations['ID'] == video_id]
        
        if not collaborative_row.empty and not content_based_row.empty:
            # Both scores are available, combine them
            collaborative_score = collaborative_row['Predicted Rating'].values[0]
            content_based_score = normalize_content_based_score(content_based_row['Similarity Score'].values[0])
            combined_score = (collaborative_score * weight_collaborative) + (content_based_score * (1 - weight_collaborative))
            recommendation_type = 'Combined'
        elif not collaborative_row.empty:
            # Only collaborative score is available
            combined_score = collaborative_row['Predicted Rating'].values[0]
            recommendation_type = 'Collaborative'
        elif not content_based_row.empty:
            # Only content-based score is available
            content_based_score = normalize_content_based_score(content_based_row['Similarity Score'].values[0])
            combined_score = content_based_score
            recommendation_type = 'Content-Based'
        else:
            # No score available for this video
            continue
        
        combined_recommendations[video_id] = {'score': combined_score, 'type': recommendation_type}
    
    # Normalize combined scores to [0, 1]

    if combined_recommendations:
        max_combined_score = max(score['score'] for score in combined_recommendations.values())
        min_combined_score = min(score['score'] for score in combined_recommendations.values())
        
        for video_id, details in combined_recommendations.items():
            if max_combined_score != min_combined_score:
                normalized_score = (details['score'] - min_combined_score) / (max_combined_score - min_combined_score)
            else:
                normalized_score = details['score']
            combined_recommendations[video_id]['score'] = normalized_score
    
    # Finalize recommendations: use combined if score is 1, otherwise use the highest individual score
    final_recommendations = {}
    
    for video_id, details in combined_recommendations.items():
        if details['score'] == 1:
            final_score = details['score']
            final_type = 'Combined'
        else:
            # Get individual scores
            collaborative_row = collaborative_recommendations[collaborative_recommendations['ID'] == video_id]
            content_based_row = content_based_recommendations[content_based_recommendations['ID'] == video_id]
            
            if not collaborative_row.empty and not content_based_row.empty:
                collaborative_score = collaborative_row['Predicted Rating'].values[0]
                content_based_score = normalize_content_based_score(content_based_row['Similarity Score'].values[0])
                
                if collaborative_score > content_based_score:
                    final_score = collaborative_score
                    final_type = 'Collaborative'
                else:
                    final_score = content_based_score
                    final_type = 'Content-Based'
            elif not collaborative_row.empty:
                final_score = collaborative_row['Predicted Rating'].values[0]
                final_type = 'Collaborative'
            elif not content_based_row.empty:
                final_score = normalize_content_based_score(content_based_row['Similarity Score'].values[0])
                final_type = 'Content-Based'
            else:
                continue
        
        final_recommendations[video_id] = {'score': final_score, 'type': final_type}
    
    # Convert final recommendations into a DataFrame

    final_recommendations_df = pd.DataFrame.from_dict(final_recommendations, orient='index').reset_index()
    final_recommendations_df.columns = ['Video ID', 'Final Score', 'Recommendation Type']
    final_recommendations_df = final_recommendations_df.sort_values(by='Final Score', ascending=False).head(top_n)
    
    return final_recommendations_df

'''
 
# featuer engineering: experiment with this to create recommendation more diverse.
# hyperparater tuning : experiment with different approach like Grid Search, Random Search,  Bayesian Optimization, Gradient-Based Optimization etc.
# hyperparameter adjusting: Consider tuning hyperparameters like the number of epochs, batch size, and learning rate based on the cross-validation results.
# parameter tuning: tune parameter according to your needs.

'''
Some popular libraries for hyperparameter tuning are:

1. Hyperopt
2. Optuna
3. Scikit-Optimize
4. Ray Tune
5. Keras Tuner

'''

'''
1.} Additional Note: this scripts contains every steps from preprocessing to model train for recommendation 
2.} The person using this needs to adjust fields according to his/her need and and must ensure exact fields
should be here according actual to the dataset to avoid keyerror.

'''

'''
to run the code you provided, you need to install the following libraries:

1. Pandas: pip install pandas
2. Scikit-learn: pip install scikit-learn
3. TensorFlow: pip install tensorflow
4. NumPy: pip install numpy (usually installed with scikit-learn or TensorFlow)

'''


'''
Optional libraries (not explicitly used in the code, but might be useful for related tasks):

1. Matplotlib and Seaborn for data visualization: pip install matplotlib seaborn
2. Jupyter Notebook for interactive development: pip install jupyter

Make sure you have Python installed (preferably the latest version) and a package manager like pip. Then, run the installation commands in your terminal or command prompt.

'''

# Sugesstion: use environment for the project, can use conda because it supports both conda and pip
'If you are using a Python environment like Conda, you can install the libraries using conda commands and pip also'
