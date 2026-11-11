import argparse
import json
from pathlib import Path

import torch

from .config import ChimeraConfig
from .data import JointDataset, collate_records, synthetic_batch
from .model import Chimera
from .tokenizer import ByteTokenizer
from .training import load_checkpoint, save_checkpoint, train


def inspect_config(config):
    # Pas d'allocation des poids, même pour une architecture de centaines de milliards.
    with torch.device("meta"):
        model = Chimera(config)
    total = sum(p.numel() for p in model.parameters())
    expert_size = 3 * config.d_model * config.hidden_dim
    unselected = config.n_layers * (config.n_experts - config.top_k) * expert_size
    return {
        "parameters": total,
        "parameters_with_selected_decoder_experts_upper_bound": total - unselected,
        "decoder_layers": config.n_layers,
        "experts_per_layer": config.n_experts,
        "selected_experts_per_token": config.top_k,
        "shared_experts_per_layer": config.shared_experts,
        "weights_gib_fp32": round(total * 4 / 2**30, 3),
        "weights_gib_bf16": round(total * 2 / 2**30, 3),
        "adamw_gib_fp32_excluding_activations": round(total * 16 / 2**30, 3),
        "modules": {name: sum(p.numel() for p in module.parameters())
                    for name, module in model.named_children() if name != "lm_head"},
    }


def get_config(args):
    return ChimeraConfig.load(args.config) if args.config else ChimeraConfig.preset(args.preset)


def main(argv=None):
    parser = argparse.ArgumentParser(description="CHIMERA — réseau multimodal à experts")
    parser.add_argument("--threads", type=int, default=4, help="Threads CPU (défaut : 4)")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Compter les paramètres sans allouer les poids")
    inspect.add_argument("--all", action="store_true", help="Inspecter tous les presets")
    demo = commands.add_parser("demo", help="Exécuter les trois modalités et les trois sorties")
    training = commands.add_parser("train", help="Entraînement conjoint réel")
    for command in (inspect, demo, training):
        command.add_argument("--preset", choices=("tiny", "small", "workstation", "base", "large", "ultra"), default="tiny")
        command.add_argument("--config", help="Configuration JSON personnalisée")
    demo.add_argument("--checkpoint")
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--output", help="Enregistrer les sorties dans ce fichier JSON")
    training.add_argument("--steps", type=int, default=100)
    training.add_argument("--stop-after", type=int, help="Arrêt propre avant la fin du planning, pour reprise")
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--accumulation", type=int, default=1)
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=0.01)
    training.add_argument("--grad-clip", type=float, default=1.0)
    training.add_argument("--warmup", type=int, default=10)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--modality-dropout", type=float, default=0.15,
                          help="Probabilité d'omettre chaque modalité pendant l'entraînement")
    training.add_argument("--device", default="cpu")
    training.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    training.add_argument("--data", help="Jeu d'entraînement JSONL ; sinon données synthétiques")
    training.add_argument("--val-data", help="Jeu de validation JSONL indépendant")
    training.add_argument("--resume", help="Checkpoint complet dont reprendre l'entraînement")
    training.add_argument("--output", default="runs/chimera")
    training.add_argument("--save-every", type=int, default=25)
    training.add_argument("--log-every", type=int, default=10)
    training.add_argument("--eval-every", type=int, default=25, help="Fréquence des validations et sélection du meilleur modèle")
    training.add_argument("--patience", type=int, default=0, help="Validations sans progrès avant arrêt ; 0 désactive l'arrêt")
    training.add_argument("--min-delta", type=float, default=0.0, help="Baisse minimale de la perte de validation")
    training.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=None)
    training.add_argument("--workers", type=int, default=0, help="Processus de chargement des données")
    generate = commands.add_parser("generate", help="Générer du texte avec le cache KV")
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--prompt", default="Bonjour")
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-k", type=int, default=40)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--device", default="cpu")
    predict = commands.add_parser("predict", help="Classer et prévoir depuis ses propres données JSONL")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--data", required=True)
    predict.add_argument("--output", help="Fichier JSONL de résultats ; sinon stdout")
    predict.add_argument("--max-new-tokens", type=int, default=0, help="Générer aussi du texte si un prompt est présent")
    predict.add_argument("--device", default="cpu")
    export = commands.add_parser("export", help="Exporter les poids sans l'optimiseur pour l'inférence")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads doit être > 0")
    torch.set_num_threads(args.threads)
    if args.command == "inspect":
        if args.all:
            report = {name: inspect_config(ChimeraConfig.preset(name))
                      for name in ("tiny", "small", "workstation", "base", "large", "ultra")}
        else:
            report = inspect_config(get_config(args))
        print(json.dumps(report, indent=2))
    elif args.command == "train":
        train(args, get_config(args))
    elif args.command == "demo":
        torch.manual_seed(42)
        model = (load_checkpoint(args.checkpoint, args.device)[0] if args.checkpoint
                 else Chimera(get_config(args)).to(args.device))
        model.eval()
        batch = synthetic_batch(model.config, batch_size=2, seed=7, device=args.device)
        with torch.inference_mode():
            output = model(**batch)
            prompt = torch.tensor([ByteTokenizer().encode("Signal", eos=False)], device=args.device)
            generated = model.generate(prompt, images=batch["images"][:1], series=batch["series"][:1],
                                       max_new_tokens=16, temperature=0)
        report = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "text_logits_shape": list(output.logits.shape),
            "classification_shape": list(output.classification_logits.shape),
            "forecast_shape": list(output.forecast.shape),
            "predicted_classes": output.classification_logits.argmax(-1).tolist(),
            "target_classes": batch["class_labels"].tolist(),
            "quantiles": list(model.config.quantiles),
            "forecast_first_example": output.forecast[0].tolist(),
            "loss": output.loss.item(),
            "losses": {key: value.item() for key, value in output.losses.items()},
            "expert_usage": output.expert_usage.tolist(),
            "generated_text": ByteTokenizer().decode(generated[0]),
            "checkpoint": args.checkpoint,
            "note": "Données synthétiques ; aucune capacité générale n'est démontrée par cette démo.",
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif args.command == "generate":
        model, _ = load_checkpoint(args.checkpoint, args.device)
        tokens = torch.tensor([ByteTokenizer().encode(args.prompt, eos=False)], device=args.device)
        generated = model.generate(tokens, max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        print(ByteTokenizer().decode(generated[0]))
    elif args.command == "predict":
        if args.max_new_tokens < 0:
            parser.error("max-new-tokens doit être >= 0")
        model, _ = load_checkpoint(args.checkpoint, args.device)
        model.eval()
        dataset = JointDataset(args.data, model.config)
        rows = []
        with torch.inference_mode():
            for index in range(len(dataset)):
                batch = {key: value.to(args.device) for key, value in collate_records([dataset[index]]).items()
                         if key not in ("labels", "class_labels", "forecast_targets")}
                output = model(**batch)
                row = {
                    "index": index,
                    "class_id": output.classification_logits[0].argmax().item(),
                    "class_probabilities": output.classification_logits[0].float().softmax(-1).tolist(),
                    "quantiles": list(model.config.quantiles),
                    "forecast": output.forecast[0].tolist(),
                }
                if args.max_new_tokens and "input_ids" in batch:
                    prompt = batch["input_ids"]
                    if prompt[0, -1] == ByteTokenizer.eos_id:
                        prompt = prompt[:, :-1]
                    context = {key: value for key, value in batch.items()
                               if key not in ("input_ids", "attention_mask")}
                    generated = model.generate(prompt, max_new_tokens=args.max_new_tokens,
                                               temperature=0, **context)
                    row["generated_text"] = ByteTokenizer().decode(generated[0])
                rows.append(json.dumps(row, ensure_ascii=False))
        content = "\n".join(rows) + "\n"
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(f"{len(rows)} prédictions écrites dans {path}")
        else:
            print(content, end="")
    elif args.command == "export":
        model, payload = load_checkpoint(args.checkpoint)
        step = payload["step"]
        del payload
        save_checkpoint(args.output, model, step=step, trainer={"inference_only": True})
        print(f"Poids d'inférence écrits dans {args.output}")


if __name__ == "__main__":
    main()
