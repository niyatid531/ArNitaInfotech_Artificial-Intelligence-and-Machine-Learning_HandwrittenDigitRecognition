import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model            import LogisticRegression
from sklearn.svm                     import LinearSVC
from sklearn.naive_bayes             import MultinomialNB
from sklearn.pipeline                import Pipeline
from sklearn.metrics                 import (classification_report,
                                              accuracy_score,
                                              confusion_matrix,
                                              roc_auc_score)
from sklearn.model_selection         import train_test_split
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/spam.csv', encoding='latin-1')
# Keep only label and message columns, rename them
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
print('Shape:', df.shape)
print('Spam rate:', round((df['label']=='spam').mean()*100, 2), '%')
print(df['label'].value_counts())

def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)   # letters only
    return re.sub(r'\s+', ' ', text).strip() # collapse spaces
df['clean_msg'] = df['message'].apply(clean)
# Encode label: ham=0, spam=1
df['label_num'] = (df['label'] == 'spam').astype(int)
print('Sample cleaned messages:')
print(df[['message', 'clean_msg', 'label']].head(3))

X = df['clean_msg']
y = df['label_num']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print('Train size:', len(X_train))
print('Test size: ', len(X_test))

tfidf = TfidfVectorizer(
    max_features = 5000,
    ngram_range  = (1, 2),   # unigrams + bigrams
    sublinear_tf = True,
)
models = {
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000,
                     ngram_range=(1,2), sublinear_tf=True)),
        ('clf',   MultinomialNB(alpha=0.1))
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', tfidf),
        ('clf',   LogisticRegression(max_iter=1000,
                     class_weight='balanced'))
    ]),
    'Linear SVM': Pipeline([
        ('tfidf', tfidf),
        ('clf',   LinearSVC(C=1.0, max_iter=2000,
                     class_weight='balanced'))
    ]),
}

scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    roc   = roc_auc_score(y_test, preds)
    scores[name] = {'accuracy': acc, 'roc_auc': roc}
    print(f'\n{name} — Accuracy: {acc:.4f} | ROC-AUC: {roc:.4f}')
    print(classification_report(y_test, preds,
          target_names=['Ham', 'Spam'], zero_division=0))
    print('-' * 60)
best_name  = max(scores, key=lambda x: scores[x]['roc_auc'])
best_model = models[best_name]
print(f'\nBest model: {best_name}')

os.makedirs('outputs', exist_ok=True)
joblib.dump(best_model, 'outputs/spam_model.pkl')
final_preds = best_model.predict(X_test)
results_df  = pd.DataFrame({
    'message':          X_test.values,
    'actual':           y_test.values,
    'predicted':        final_preds,
    'actual_label':     ['spam' if x==1 else 'ham' for x in y_test.values],
    'predicted_label':  ['spam' if x==1 else 'ham' for x in final_preds]
})
results_df.to_csv('outputs/predictions.csv', index=False)
print('Saved: outputs/spam_model.pkl')
print('Saved: outputs/predictions.csv')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Plot 1: Ham vs Spam distribution
counts = df['label'].value_counts()
axes[0].bar(counts.index, counts.values,
            color=['#2455A4', '#C0392B'])
axes[0].set_title('Ham vs Spam Distribution', fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')
# Plot 2: Message length distribution
df['msg_len'] = df['message'].apply(len)
axes[1].hist(df[df['label']=='ham']['msg_len'],  bins=50,
             alpha=0.6, label='Ham',  color='#2455A4')
axes[1].hist(df[df['label']=='spam']['msg_len'], bins=50,
             alpha=0.6, label='Spam', color='#C0392B')
axes[1].set_title('Message Length by Label', fontweight='bold')
axes[1].set_xlabel('Character Count')
axes[1].legend()
# Plot 3: Model ROC-AUC comparison
model_names = list(scores.keys())
roc_scores  = [scores[m]['roc_auc'] for m in model_names]
axes[2].bar(model_names, roc_scores,
            color=['#27AE60', '#2455A4', '#E67E22'])
axes[2].set_title('Model ROC-AUC Comparison', fontweight='bold')
axes[2].set_ylabel('ROC-AUC Score')
axes[2].set_ylim(0.8, 1.0)
for i, v in enumerate(roc_scores):
    axes[2].text(i, v + 0.002, f'{v:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots.png', dpi=150)
plt.close()
print('Plot saved to outputs/plots.png')

def predict_sms(message: str) -> str:
    cleaned = clean(message)
    pred    = best_model.predict([cleaned])[0]
    return 'SPAM' if pred == 1 else 'HAM (Legitimate)'
if __name__ == '__main__':
    messages = [
        'WINNER! You have been selected for a $1000 cash prize. Call NOW!',
        'Hey, are we still meeting for lunch tomorrow?',
        'Urgent! Your account has been suspended. Click here to verify.',
        'Can you pick up some milk on your way home?',
    ]
    for msg in messages:
        print(f'{predict_sms(msg):18s}  <--  {msg[:55]}')

