from models import *
from Augmentations import *
from DataLoader import *
from utils import *
from criterion import *
import pandas as pd
import argparse
from torch.autograd import Variable
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--encoder", type=str, default="b2")
parser.add_argument("--model", type=str, default="Unet")
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--kfold", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--model_checkpoint", type=str, default="")
parser.add_argument("--max_epoch", type=int, default=50)
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--df_train_path", type=str, default="train.csv")
parser.add_argument("--df_test_path", type=str, default="sample_submission.csv")
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

# Seed everything
SEED = args.seed

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

np.random.seed(SEED)
random.seed(SEED)


def _worker_init_fn(worker_id):
    np.random.seed(worker_id)


# KFOLD split
def split_train_valid(fold_to_train=0):
    labels = []
    for i in tqdm_notebook(range(train_df.ImageId.nunique())):
        im_name = train_df.iloc[4 * i].ImageId
        labels.append(
            train_df[train_df.ImageId == im_name]["hasMask"].values.astype(int)
        )

    str_labels = ["".join([str(x) for x in labels[k]]) for k in range(len(labels))]

    le = preprocessing.LabelEncoder()
    le.fit(str_labels)
    new_labels = le.transform(str_labels)

    indices = [i for i in range(len(new_labels))]

    skf = StratifiedKFold(n_splits=5, random_state=SEED)

    for fold, (train_index, test_index) in enumerate(skf.split(indices, new_labels)):
        if fold == fold_to_train:
            break

    trn_indexes = []
    val_indexes = []

    for k, i in enumerate(train_index):
        train_index[k] = 4 * i

    for k, i in enumerate(test_index):
        test_index[k] = 4 * i

    return np.array(train_index), np.array(test_index)


Global_fold = args.fold

trn_indexes, val_indexes = split_train_valid(fold_to_train=Global_fold)


# Create dataset objects
train_dataset = Dataset(
    "train_images/",
    train_df.iloc[trn_indexes],
    transform=train_transforms,
    t_mode="train",
)
valid_dataset = Dataset(
    "train_images/",
    train_df.iloc[val_indexes],
    transform=test_transforms,
    t_mode="valid",
)

device = torch.device(args.device)
enc_name = args.encoder
model = Model(enc_name).to(device)

optimizer = torch.optim.Adam(model.parameters(), 1e-4)

scheduler_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=1 / np.sqrt(10), threshold=0, patience=2, verbose=True
)

scheduler_warmup = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=1, gamma=np.sqrt(10)
)

if args.model_checkpoint:
    load_checkpoint(args.model_checkpoint, model, optimizer)

start_epoch = args.start_epoch
NUM_EPOCHS = args.max_epoch
BATCH_SIZE = args.batch_size
WARMUP_epoch = 2
best_score = -1

for epoch in range(start_epoch, NUM_EPOCHS):

    print("Epoch : ", epoch)

    # Train
    train_loss = []
    model.train()

    for image, masks in tqdm_notebook(
        data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            worker_init_fn=_worker_init_fn,
        ),
        disable=True,
    ):

        image = image.type(torch.FloatTensor).to(device)

        y_pred = model(Variable(image))

        loss = loss_bce(y_pred, Variable(masks.to(device)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    if epoch < 4:
        scheduler_warmup.step()
        continue

    # Validation
    model.eval()
    unet_score = 0

    for image, masks in tqdm_notebook(
        data.DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            worker_init_fn=_worker_init_fn,
        ),
        disable=True,
    ):

        image = image.type(torch.FloatTensor).to(device)

        y_pred = model(Variable(image))
        y_pred = torch.sigmoid(y_pred) > 0.5

        for s in range(y_pred.shape[0]):

            sample_score = 0

            for channel in range(4):

                if (
                    y_pred[s, channel, :, :].cpu().data.numpy().astype(int)
                ).sum() < 512:

                    sample_score += (
                        dice(
                            masks[s, channel, :, :].cpu().data.numpy().astype(int),
                            np.zeros((320, 480)),
                        )
                        / 4
                    )

                else:

                    sample_score += (
                        dice(
                            y_pred[s, channel, :, :].cpu().data.numpy().astype(int),
                            masks[s, channel, :, :].cpu().data.numpy().astype(int),
                        )
                        / 4
                    )

            unet_score += sample_score

    unet_score /= len(valid_dataset)

    if unet_score > best_score:
        save_checkpoint(
            "{}_fold{}.pth".format(enc_name, Global_fold, epoch), model, optimizer
        )
        best_score = unet_score

    print("Loss : {} , Validation : {}".format(np.mean(train_loss), unet_score))

    scheduler_decay.step(unet_score)
