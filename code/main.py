from utils import prepare_data
from My_Dataset import My_Dataset
import os 
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Encoder import Encoder_Sentiment_Analysis
from tqdm import tqdm

# HYPER PARAMETERS:
input_dim = 100
batch_size = 128
num_heads = 10
lr=7e-5
max_epochs = 10

# will have to break down the dataset into smaller parts, memory is exceeding when trying to process the whole dataset at once

device = torch.device("mps")


# let's try to train the model




if __name__ == "__main__":

    # raw_data = My_Dataset()
    # prepare_data(raw_data, input_dim)
    # sys.exit()

    data = torch.load('my_dataset_max_len_64_input_dim_100.pt')
    train_loader = DataLoader(data, batch_size, shuffle=True)

    model = Encoder_Sentiment_Analysis(input_dim, num_heads)
    model.load_state_dict(torch.load('test_set_sub_1/sentiment.params'))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    loss_list = []

    for epoch in tqdm(range(max_epochs)):

        print('*' * 100)
        print('\n\n')
        print(f"epoch = {epoch}/{max_epochs}")

        epoch_loss = 0.0

        for step, temp in tqdm(enumerate(train_loader)):
            # returns a list of length 3
            lemma_tensors = temp[0]
            pos_tensors = torch.stack(temp[1], dim=1)
            label_tensors = temp[2]

            # print(type(lemma_tensors))
            # print(type(pos_tensors))
            # print(type(label_tensors))

            # sys.exit()


            # fwd_pass
            output = model([lemma_tensors.to(device), pos_tensors.to(device)])
            loss = criterion(output, label_tensors.to(device))

            # print(f"Step = {step}/{len(train_loader)}, step_loss = {loss.item()}")

            epoch_loss += loss.item()

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # update the params
            optimizer.step()

        print(f"epoch = {epoch}/{max_epochs}, epoch_loss = {epoch_loss}")
        print('\n\n')
        print('*' * 100)

        with open("loss.txt_mps", "a") as f: 
            my_dict = {'epoch' : epoch, 'max_epochs': max_epochs, 'epoch_loss': epoch_loss}
            f.write(f"{my_dict}\n")
        
        loss_list.append(epoch_loss)

        torch.save(model.state_dict().cpu(), 'sentiment.params_mps')
        



    