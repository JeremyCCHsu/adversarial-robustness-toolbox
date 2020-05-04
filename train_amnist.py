from argparse import ArgumentParser
import logging
import time

import torch

from dev.loaders import AudioMNISTDataset, PreprocessRaw
from dev.models import RawAudioCNN
from dev.transforms import Preprocessor

# set global variables
AUDIO_DATA_TRAIN_ROOT = "data/audiomnist/train"
AUDIO_DATA_TEST_ROOT = "data/audiomnist/test"

n_epochs = 50
epsilon = .0005

def _is_cuda_available():
    return torch.cuda.is_available()


def _get_device():
    return torch.device("cuda" if _is_cuda_available() else "cpu")


def main(args):
    # Step 0: parse args and init logger
    logging.basicConfig(level=logging.INFO)

    generator_params = {
        'batch_size': 64,
        'shuffle': True,
        'num_workers': 6
    }

    # Step 1: load data set
    train_data = AudioMNISTDataset(
        root_dir=AUDIO_DATA_TRAIN_ROOT,
        transform=PreprocessRaw(),
    )
    test_data = AudioMNISTDataset(
        root_dir=AUDIO_DATA_TEST_ROOT,
        transform=PreprocessRaw(),
    )

    train_generator = torch.utils.data.DataLoader(
        train_data,
        **generator_params,
    )
    test_generator = torch.utils.data.DataLoader(
        test_data,
        **generator_params,
    )

    # prep = Preprocessor()

    # Step 2: prepare training
    device = _get_device()
    logging.info(device)

    model = RawAudioCNN(num_class=10)
    if _is_cuda_available():
        model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Step 3: train
    for epoch in range(n_epochs):
        # training loss
        training_loss = 0.0
        # validation loss
        validation_loss = 0
        # accuracy
        correct = 0
        total = 0

        model.train()
        for batch_idx, batch_data in enumerate(train_generator):

            inputs = batch_data['input']  # [B, c=1, T]
            labels = batch_data['digit']

            # # ======== Augmentation ===============
            # a = 2 * torch.rand([])
            # noisy = inputs + (torch.rand_like(inputs) * a * 2 * epsilon - a * epsilon)
            # inputs = torch.cat([inputs, noisy])
            # labels = torch.cat([labels, labels])
            # # ======== ============ ===============

            if _is_cuda_available():
                inputs = inputs.to(device)
                labels = labels.to(device)
            # Model computations
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # sum training loss
            training_loss += loss.item()
            
        model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_generator):
                inputs = batch_data['input']
                labels = batch_data['digit']
                if _is_cuda_available():
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # sum validation loss
                validation_loss += loss.item()
                # calculate validation accuracy
                predictions = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        # calculate final metrics
        validation_loss /= len(test_generator)
        training_loss /= len(train_generator)
        accuracy = 100 * correct / total
        logging.info(f"[Ep {epoch+1:02d}/{n_epochs}] train-loss: {training_loss:.3f}"
                     f"\tval-loss: {validation_loss:.3f}"
                     f"\taccuracy: {accuracy:.2f}")

        # Checkpointing
        torch.save(
            model,
            f"model/model_raw_audio_tmp.pt"
        )


    logging.info("Finished Training")

    # Step 4: save model
    if args.model_ckpt is None:
        ckpt = f"model/model_raw_audio_{time.strftime('%Y%m%d%H%M')}.pt"
    else:
        ckpt = args.model_ckpt
    torch.save(model, ckpt)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_ckpt", type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
