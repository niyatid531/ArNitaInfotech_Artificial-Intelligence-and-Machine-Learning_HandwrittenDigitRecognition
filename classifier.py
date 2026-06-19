import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data=datasets.MNIST('data', train=True,  download=True, transform=transform)
test_data=datasets.MNIST('data', train=False, download=True, transform=transform)

train_loader=DataLoader(train_data, batch_size=128, shuffle=True)
test_loader=DataLoader(test_data,  batch_size=128, shuffle=False)

print(f'Train: {len(train_data)} images')
print(f'Test:  {len(test_data)}  images')

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model=CNN().to(device)
optimizer=optim.Adam(model.parameters(), lr=0.001)
criterion=nn.CrossEntropyLoss()

def train_epoch(model, loader):
    model.train()
    total_loss, correct=0, 0
    for images, labels in loader:
        images, labels=images.to(device), labels.to(device)
        optimizer.zero_grad()
        out=model(images)
        loss=criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        correct+=(out.argmax(1)==labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def eval_epoch(model, loader):
    model.eval()
    total_loss, correct=0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels=images.to(device), labels.to(device)
            out=model(images)
            loss=criterion(out, labels)
            total_loss+=loss.item()
            correct+=(out.argmax(1)==labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


history={'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
best_val_acc, patience, patience_count=0, 3, 0

print('\nTraining...')
for epoch in range(1, 16):
    tr_loss, tr_acc=train_epoch(model, train_loader)
    va_loss, va_acc=eval_epoch(model, test_loader)

    history['train_acc'].append(tr_acc)
    history['val_acc'].append(va_acc)
    history['train_loss'].append(tr_loss)
    history['val_loss'].append(va_loss)

    print(f'Epoch {epoch:2d}/15 — loss: {tr_loss:.4f} acc: {tr_acc:.4f} '
          f'| val_loss: {va_loss:.4f} val_acc: {va_acc:.4f}')

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        torch.save(model.state_dict(), 'outputs/best_weights.pt')
        patience_count=0
    else:
        patience_count+=1
        if patience_count>=patience:
            print(f'Early stopping at epoch {epoch}')
            break

model.load_state_dict(torch.load('outputs/best_weights.pt', weights_only=True))

_, test_acc=eval_epoch(model, test_loader)
print(f'\nTest Accuracy: {test_acc*100:.2f}%')

all_preds, all_labels=[], []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        preds=model(images.to(device)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds=np.array(all_preds)
all_labels=np.array(all_labels)

print('\nClassification Report:')
print(classification_report(all_labels, all_preds,
      target_names=[str(i) for i in range(10)]))

os.makedirs('outputs', exist_ok=True)
torch.save(model.state_dict(), 'outputs/digit_model.pt')
pd.DataFrame({'actual': all_labels, 'predicted': all_preds}) \
  .to_csv('outputs/predictions.csv', index=False)
print('Saved: outputs/digit_model.pt')
print('Saved: outputs/predictions.csv')

fig, axes=plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history['train_acc'], label='Train', color='#2455A4')
axes[0].plot(history['val_acc'],   label='Val',   color='#27AE60')
axes[0].set_title('Accuracy Over Epochs', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend()

axes[1].plot(history['train_loss'], label='Train', color='#C0392B')
axes[1].plot(history['val_loss'],   label='Val',   color='#E67E22')
axes[1].set_title('Loss Over Epochs', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend()

cm=confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=range(10), yticklabels=range(10))
axes[2].set_title('Confusion Matrix', fontweight='bold')
axes[2].set_ylabel('Actual'); axes[2].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('outputs/plots.png', dpi=150)
plt.close()
print('Saved: outputs/plots.png')

images_sample, labels_sample = next(iter(
    DataLoader(test_data, batch_size=10, shuffle=True)
))
preds_sample = model(images_sample.to(device)).argmax(1).cpu().numpy()

fig, axes=plt.subplots(2, 5, figsize=(12, 5))
for i in range(10):
    ax=axes[i//5, i%5]
    ax.imshow(images_sample[i].squeeze(), cmap='gray')
    actual=labels_sample[i].item()
    predicted=preds_sample[i]
    color='green' if actual==predicted else 'red'
    ax.set_title(f'A:{actual} P:{predicted}', color=color, fontweight='bold')
    ax.axis('off')

plt.suptitle('Sample Predictions — Green=Correct  Red=Wrong', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/sample_predictions.png', dpi=150)
plt.close()
print('Saved: outputs/sample_predictions.png')

def predict_digit(image_tensor):
    model.eval()
    with torch.no_grad():
        out=model(image_tensor.unsqueeze(0).to(device))
        probs=torch.softmax(out, dim=1)[0]
        digit=probs.argmax().item()
        return digit, probs[digit].item() * 100


if __name__ == '__main__':
    print('\nSample predictions:')
    for _ in range(5):
        idx=np.random.randint(len(test_data))
        img, label=test_data[idx]
        digit, conf=predict_digit(img)
        status= 'CORRECT' if digit==label else 'WRONG'
        print(f'  Actual: {label} | Predicted: {digit} | '
              f'Confidence: {conf:.1f}% | {status}')
