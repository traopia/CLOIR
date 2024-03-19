import torch
from create_data_loader import TripletLossDataset_features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
from Triplet_Network import TripletResNet_features
import glob

def get_inputs(data, feature):
    positive_examples_list = []
    negative_examples_list = []

    positive = data.positive_examples
    negative = data.negative_examples
    for i in range(len(data.df)):
        reference_feature = data.df.loc[i, feature]
        positive_features = data.df.loc[[j for j in positive[i]],feature]
        negative_features = data.df.loc[[j for j in negative[i]],feature]
        positive_examples = torch.stack([torch.cat((feat, reference_feature), dim=0) for feat in positive_features])
        negative_examples = torch.stack([torch.cat((feat, reference_feature), dim=0) for feat in negative_features])
        positive_examples_list.append(positive_examples)
        negative_examples_list.append(negative_examples)

    positive_examples_all = torch.stack(positive_examples_list)
    negative_examples_all = torch.stack(negative_examples_list)

    return positive_examples_all, negative_examples_all


def model_training(positive_examples_all, negative_examples_all, model_path):
    # Splitting the data into training and testing sets

    X_positive = positive_examples_all.reshape(-1, 1024)
    X_negative = negative_examples_all.reshape(-1, 1024)
    X = torch.cat((X_positive, X_negative), dim=0)
    y = torch.cat((torch.ones(X_positive.shape[0]), torch.zeros(X_negative.shape[0])))

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)  # Assuming y contains integer labels

    # Sending tensors to a specific device (e.g., GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    X_tensor = X_tensor.to(device)
    y_tensor = y_tensor.to(device)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.cpu().numpy())
    X_test_scaled = scaler.transform(X_test.cpu().numpy())
    # Initializing the logistic regression model
    model = LogisticRegression(max_iter = 1000)

    # Training the model on the training data
    model.fit(X_train_scaled, y_train.cpu().numpy())  # LogisticRegression expects numpy arrays

    # Predicting labels for the test set
    y_pred = model.predict(X_test_scaled)  # LogisticRegression expects numpy arrays

    # Calculating the accuracy of the model
    accuracy = accuracy_score(y_test.cpu().numpy(), y_pred)

    print("Accuracy:", accuracy)
    model_name = model_path+'/binary_classifier.pkl'
    with open(model_name, 'wb') as f:
        pickle.dump(model, f)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    data_path = 'DATA/Dataset_toload/train_dataset_image_features_posfaiss_negfaiss.pt'
    data = torch.load(data_path)
    feature = 'image_features'
    model = TripletResNet_features(data.df.loc[0,feature].shape[0])
    # trained_models_path = glob('trained_models/*', recursive = True)
    # for i in trained_models_path:
    #     if feature in i:
    #         print(f'Features with model {i}')
    #         model_path = i + '/model.pth'
    name =  '_'.join(data_path.split('/')[-1].split('.')[0].split('_')[2:])
    model_path = f'trained_models/TripletResNet_{name}'
    model.load_state_dict(torch.load(model_path+'/model.pth', map_location=torch.device('cpu')))
    model.eval()
    data.df[f'trained_{feature}'] = data.df[feature].apply(lambda x: model.forward_once(x).detach())

    positive_examples_all, negative_examples_all = get_inputs(data,f'trained_{feature}')
    model_training(positive_examples_all, negative_examples_all,model_path)



    
if __name__ == '__main__':
    main()