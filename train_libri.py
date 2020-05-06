from argparse import ArgumentParser
import logging
import time

import torch

from dev.loaders import LibriSpeech4SpeakerRecognition
from dev.models import RawAudioCNN


# set global variables
n_epochs = 50   # FIXME: `num_ites` is a better indicator
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
        'num_workers': 8
    }

    # Step 1: load data set
    train_data = LibriSpeech4SpeakerRecognition(
        root=args.data_root, 
        url=args.set,
        train_speaker_ratio=1,
        train_utterance_ratio=0.9,
        subset="train",
        project_fs=16000,
        wav_length=args.wav_length,
    )
    train_generator = torch.utils.data.DataLoader(
        train_data,
        **generator_params,
    )
  
    # test_data = AudioMNISTDataset(
    #     root_dir=AUDIO_DATA_TEST_ROOT,
    #     transform=PreprocessRaw(),
    # )

    # test_generator = torch.utils.data.DataLoader(
    #     test_data,
    #     **generator_params,
    # )

    # Step 2: prepare training
    device = _get_device()
    logging.info(device)

    model = RawAudioCNN(num_class=251)
    if _is_cuda_available():
        model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-3, momentum=0.9)

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
        for batch_idx, batch_data in enumerate(train_generator, 1):

            # inputs = batch_data['input']  # [B, c=1, T]
            inputs, labels = batch_data

            # ======== Augmentation ===============
            a = torch.rand([])
            noise = 2 * a * args.epsilon * torch.rand_like(inputs) - a * args.epsilon
            noisy = inputs + noise
            inputs = torch.cat([inputs, noisy])
            labels = torch.cat([labels, labels])
            # ======== ============ ===============

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

            print(f"[Ep {epoch+1:02d}/{n_epochs}] It [{batch_idx}] train-loss: {training_loss / batch_idx:.4f}", end="\r")

            
        # model.eval()
        # with torch.no_grad():
        #     for batch_idx, batch_data in enumerate(test_generator):
        #         inputs = batch_data['input']
        #         labels = batch_data['digit']
        #         if _is_cuda_available():
        #             inputs = inputs.to(device)
        #             labels = labels.to(device)
        #         outputs = model(inputs)
        #         loss = criterion(outputs, labels)
        #         # sum validation loss
        #         validation_loss += loss.item()
        #         # calculate validation accuracy
        #         predictions = torch.max(outputs.data, 1)[1]
        #         total += labels.size(0)
        #         correct += (predictions == labels).sum().item()

        # # calculate final metrics
        # validation_loss /= len(test_generator)
        # training_loss /= len(train_generator)
        # accuracy = 100 * correct / total
        # logging.info(f"[Ep {epoch+1:02d}/{n_epochs}] train-loss: {training_loss:.3f}"
        #              f"\tval-loss: {validation_loss:.3f}"
        #              f"\taccuracy: {accuracy:.2f}")

        # Checkpointing
        torch.save(
            model,
            f"model/model_raw_audio_tmp.pt"
        )
        print()


    logging.info("Finished Training")

    # Step 4: save model
    if args.model_ckpt is None:
        ckpt = f"model/libri_model_raw_audio_{time.strftime('%Y%m%d%H%M')}.pt"
    else:
        ckpt = args.model_ckpt

    torch.save(model, ckpt)


def parse_args():
    parser = ArgumentParser("Speaker Classification model on LibriSpeech dataset")
    parser.add_argument("-m", "--model_ckpt", type=str, default=None)
    parser.add_argument(
        "-d", "--data_root", type=str, required=True, 
        help="Parent directory of LibriSpeech")
    parser.add_argument(
        "-s", "--set", 
        choices=['train-clean-100', 'train-clean-360', 'train-other-500'])
    parser.add_argument(
        "-e", "--epsilon", type=float, default=0,
        help="noise magnitude in data augmentation")
    parser.add_argument(
        "-l", "--wav_length", type=int, default=80_000,
        help="max length of waveform in a batch")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
