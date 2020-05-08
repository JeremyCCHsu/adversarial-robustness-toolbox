from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from pathlib import Path

from art.classifiers import PyTorchClassifier
from art import attacks

from dev.loaders import LibriSpeech4SpeakerRecognition, LibriSpeechSpeakers
from dev.transforms import Preprocessor

from hparams import hp


# set seed
np.random.seed(123)

device = "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_attacker_args(args, eps, eps_step):
    targeted = False if args.target is None else True
    if args.attack == "DeepFool":
        kwargs = {"epsilon": eps}
    elif args.attack == "NoiseAttack":
        kwargs = {"eps": eps}
    elif args.attack in ["FastGradientMethod", "ProjectedGradientDescent"]:
        kwargs = {"eps": eps, "eps_step": eps_step, "targeted": targeted}
    else:
        raise NotImplementedError
    return kwargs


class AttackResultCounter():
    def __init__(self):
        self.n_instance = 0
        self.n_nontarget_instance = 0
        self.n_correct_prediction = 0
        self.n_successful_untargeted = 0
        self.n_successful_targeted = 0
    
    def update(self, label, prediction, adversarial_prediction, target=None):
        """ FIXME: we need to exclude the target class when calculating TASR """
        if prediction == label:
            self.n_correct_prediction += 1

        if prediction != adversarial_prediction:
            self.n_successful_untargeted += 1

        self.n_instance += 1

        if target is not None:
            if label != target:
                self.n_nontarget_instance += 1
                if adversarial_prediction == target:
                    self.n_successful_targeted += 1
    
    def asr(self):
        return self.n_successful_untargeted / self.n_instance

    def tasr(self):
        if self.n_nontarget_instance == 0:
            return np.nan
        else:
            return self.n_successful_targeted / self.n_nontarget_instance

    def accuracy(self):
        return self.n_correct_prediction / self.n_instance


class ProjectorTSVWriter():
    def __init__(self, output_dir, resolver):
        self.meta = open(output_dir / "meta.tsv", "w")
        self.meta.write(
            "\t".join(["Index", "TrueGender", "Prediction", "PredictedGender", "Source"]) + "\n"
        )

        self.vec = open(output_dir / "vec.tsv", "w")
        self.resolver = resolver

    def update(self, embedding, adv_embedding, label, pred, pred_adv, i):
        stem = f"{i:04d}-{label:03d}\t{self.resolver.get_gender(label)}"

        self.vec.write("\t".join([f"{x.item():.8f}" for x in embedding]) + "\n")
        self.meta.write(
            f"{stem}\t{pred:03d}\t{self.resolver.get_gender(pred)}\toriginal\n")

        self.vec.write("\t".join([f"{x.item()}" for x in adv_embedding]) + "\n")
        self.meta.write(
            f"{stem}\t{pred_adv:03d}\t{self.resolver.get_gender(pred_adv)}\tadversarial\n")

    def close(self):
        self.meta.close()
        self.vec.close()


class AsyncReporter():
    def __init__(self, report_file, args, counter, num_params):
        if not Path(report_file).exists():
            header = (
                f"Attacker={args.attack}\n"
                f"Accuracy={counter.accuracy(): .4f}\n"
                f"Model={args.model_ckpt} (#parameters={num_params / 1e6:.3f} million)\n"
                "| avg eps    | avg SNR   |  ASR    |\n"
                "| ------     |   ---     |  ---    |\n"
            )
            with open(args.report, "a") as fp:
                fp.write(header)

        self.report_file = report_file
    
    def update(self, average_epsilon, average_snr, rate):
        with open(self.report_file, "a") as fp:
            row = (
                f"| {average_epsilon:.4e} |   "
                f"{average_snr:5.2f}   | "
                f"{rate: .4f} |\n"
            )
            fp.write(row)


def main(args):
    # load AudioMNIST test set
    resolver = LibriSpeechSpeakers(hp.data_root, hp.data_subset)
    dataset = LibriSpeech4SpeakerRecognition(
        root=hp.data_root,
        url=hp.data_subset,
        subset="test",
        train_speaker_ratio=hp.train_speaker_ratio,
        train_utterance_ratio=hp.train_utterance_ratio,
        project_fs=hp.sr,  # FIXME: unused
        wav_length=None,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)


    # load pretrained model
    model = (
        torch.load(args.model_ckpt)
        .eval()
        .to(device)
    )


    # wrap model in a ART classifier
    classifier_art = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=None,
        input_shape=[1, 5 * hp.sr],  # FIXME
        nb_classes=resolver.get_num_speakers(),
    )

    counter = AttackResultCounter()
    tsv_writer = ProjectorTSVWriter(args.output_dir, resolver=resolver)

    snrs = []
    epss = []
    for i, (waveform, label) in enumerate(loader, 1):
        if args.target is None:
            y = None
        else:
            y = 0 * label.numpy() + args.target

        label = label.item()

        if args.epsilon is not None:
            eps = args.epsilon
        else:
            eps = (waveform.pow(2).mean() / np.power(10, args.snr / 10)).sqrt().item()

        kwargs = resolve_attacker_args(args, eps, eps_step=eps / 5)    # TODO ad-hoc
        
        # # craft adversarial example with PGD
        attacker = getattr(attacks, args.attack)(classifier_art, **kwargs)
        adv_waveform = torch.from_numpy(
            attacker.generate(waveform, y=y)
        )

        noise = adv_waveform - waveform

        snr = 10 * (waveform.pow(2).mean() / noise.pow(2).mean()).log10()
        snrs.append(snr)
        epss.append(eps)

        # evaluate the classifier on the adversarial example
        with torch.no_grad():
            emb = model.encode(waveform.to(device))
            pred = (
                model.predict_from_embeddings(emb)
                .argmax(-1)
                .tolist()[0]
            )

            adv_emb = model.encode(adv_waveform.to(device))
            pred_adv = (
                model.predict_from_embeddings(adv_emb)
                .argmax(-1)
                .tolist()[0]
            )

        counter.update(label, pred, pred_adv, target=args.target)
        tsv_writer.update(emb[0], adv_emb[0], label, pred, pred_adv, i)

        print(
            (
                f"Processing {i}/{len(loader)}: [L={label:3d}|P={pred:3d}|A={pred_adv:3d}], "
                f"SNR={snr:.2f} dB, "
                f"NASR={counter.asr():.4f}, "
                f"TASR={counter.tasr():.4f}, "
                f"accuracy={counter.accuracy():.4f}"
            ),
            end="\r"
        )
        
        if args.save_wav:
            write(
                args.output_dir / f"adv-{i:04d}-true-{label:03d}--prediction-{pred_adv:03d}.wav",
                hp.sr,
                adv_waveform[0, 0].detach().cpu().numpy()
            )

    tsv_writer.close()
    print()


    # # =====================================================
    prep = Preprocessor()
    mel = prep(waveform[0])
    nel = prep(noise[0])
    ael = prep(adv_waveform[0])

    fig, ax = plt.subplots(3, 1)
    ax[0].set_title(
        f"Truth: {label}. Prediction: {pred}. Corrupted prediction: {pred_adv}\n"
        "Top: clean. Middle: noise. Bottom: noisy"
    )
    im = ax[0].imshow(mel[0].detach().cpu().numpy(), origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax[0])
    im = ax[1].imshow(nel[0].detach().cpu().numpy(), origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax[1])
    im = ax[2].imshow(ael[0].detach().cpu().numpy(), origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax[2])
    fig.savefig(args.output_dir / f"mel.png")

    fig, ax = plt.subplots(3, 1)
    im = ax[0].plot(waveform[0, 0].detach().cpu().numpy())
    im = ax[1].plot(noise[0, 0].detach().cpu().numpy())
    im = ax[2].plot(adv_waveform[0, 0].detach().cpu().numpy())
    fig.savefig(args.output_dir / f"wav.png")

    write(args.output_dir / f"noise.wav", hp.sr, noise[0, 0].detach().cpu().numpy())
    # write(args.output_dir / "ori.wav", hp.sr, waveform[0].numpy())
    # write(args.output_dir / "adv.wav", hp.sr, adv_waveform[0].numpy())

    if args.report is not None:
        reporter = AsyncReporter(
            args.report, args, counter, 
            num_params=count_parameters(model),
        )
        reporter.update(
            average_epsilon=np.mean(epss), 
            average_snr=np.mean([s for s in snrs if s < 100]),
            rate=counter.tasr() if args.target is not None else counter.asr(),
        )
    # # =====================================================


def parse_args():
    parser = ArgumentParser("Speaker Classification model on LibriSpeech dataset")
    parser.add_argument("-m", "--model_ckpt", required=True)
    parser.add_argument("-o", "--output_dir", type=Path, default=None, required=True)
    parser.add_argument("-a", "--attack", choices=[x for x in dir(attacks) if x[0] != "_"], default="FastGradientMethod")
    parser.add_argument("-e", "--epsilon", type=float, default=None)
    parser.add_argument("-s", "--snr", type=float, default=None, help="signal-to-noise ratio (in decibel)")
    parser.add_argument(
        "-t", "--target", type=int, default=None,
        help="Attack target. Set it to `None` for untargeted attacks.")
    parser.add_argument(
        "-r", "--report", default=None,
        help="a text file for documenting the final results.")
    parser.add_argument(
        "-w", '--save_wav', dest='save_wav', action='store_true', default=False)
    args = parser.parse_args()

    assert not (args.epsilon is None and args.snr is None), "Set either `epsilon` or `snr`"

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


if __name__ == "__main__":
    main(parse_args())
