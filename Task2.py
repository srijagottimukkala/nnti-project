# import dependencies
import torch
import sklearn
import datasets
import numpy as np
import transformers
import pandas as pd
from tqdm import tqdm


ext_data = pd.read_csv("C:\\Users\\Misbahsayeeda\\Desktop\\nnti-project\\External-Dataset_for_Task2.csv")



# Load tokenizer and model


# Define dataset class
class ExternalSMILESDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data.iloc[idx]["smiles"]
        target = self.data.iloc[idx]["exp"]
        inputs = self.tokenizer(smiles, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}
        return {**inputs, "labels": torch.tensor(target, dtype=torch.float)}

# Prepare dataset and dataloader
external_dataset = ExternalSMILESDataset(ext_data, tokenizer)
external_loader = DataLoader(external_dataset, batch_size=16, shuffle=False)

# Define function to compute gradients
def compute_gradients(model, batch, criterion):
    model.zero_grad()
    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    loss = criterion(outputs.squeeze(), batch['labels'])
    loss.backward()
    gradients = [param.grad.clone() for param in model.parameters() if param.grad is not None]
    return gradients

# Define LiSSA approximation for inverse Hessian-vector product (iHVP)
def lissa_approximation(model, train_loader, test_grad, damping=0.01, num_samples=100):
    ihvp = [torch.zeros_like(g) for g in test_grad]
    criterion = nn.MSELoss()

    for _, batch in zip(range(num_samples), train_loader):
        train_grad = compute_gradients(model, batch, criterion)
        v = test_grad
        for _ in range(10):
            hvp = torch.autograd.grad(outputs=train_grad, inputs=model.parameters(), grad_outputs=v, retain_graph=True)
            v = [v_i - damping * hvp_i for v_i, hvp_i in zip(v, hvp)]
        ihvp = [ihvp_i + v_i for ihvp_i, v_i in zip(ihvp, v)]

    ihvp = [i / num_samples for i in ihvp]
    return ihvp

# Compute influence scores
criterion = nn.MSELoss()
influence_scores = []

for batch in tqdm(external_loader, desc="Computing Influence Scores"):
    ext_grad = compute_gradients(model, batch, criterion)
    ihvp = lissa_approximation(model, external_loader, ext_grad)
    influence = sum([(g * i).sum().item() for g, i in zip(ext_grad, ihvp)])
    influence_scores.append(influence)

# Add influence scores to the external dataset
ext_data['influence_score'] = influence_scores

# Select top-k influential samples
top_k = 500
selected_data = ext_data.nlargest(top_k, 'influence_score')

# Save the selected dataset
selected_data.to_csv("C:\\Users\\Misbahsayeeda\\Desktop\\nnti-project\\Selected-External-Dataset.csv", index=False)

print("Top-k influential samples selected and saved.")


