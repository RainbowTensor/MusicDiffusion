import yaml
import torch
from torch.utils.data import DataLoader
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
import wandb
from PIL import Image
import numpy as np
import os

from data.pypianoroll_dataset import PypianorollLMDB
from autoencoder.autoencoder import Autoencoder


with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)

if not os.path.isdir("models"):
    os.mkdir("models")

autoencoder_config = config["autoencoder"]
train_config = autoencoder_config["train"]
logger_kwargs = train_config["logger_kwargs"]

batch_size = train_config["batch_size"]
data_path = train_config["data_path"]
num_epochs = train_config["num_epochs"]
learning_rate = train_config["learning_rate"]
mixed_precision = train_config["mixed_precision"]
gradient_accumulation_steps = train_config["gradient_accumulation_steps"]
save_path = f'./{train_config["save_dir"]}/autoencoder.pt'

train_dataset = PypianorollLMDB(data_path)
train_dataset[0]
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)

model = Autoencoder(autoencoder_config)

total_params = sum(p.numel() for p in model.parameters())

print(f"Model has {total_params} params")


if train_config["resume"]:
    assert logger_kwargs["id"] is not None, \
        "When resuming training, WandB run ID should be set."

    run = wandb.init(
        config=config,
        project=logger_kwargs["project"],
        id=logger_kwargs["id"],
        resume="allow"
    )

    model_artifact_name = f'model-{run.id}:latest'
    artifact = run.use_artifact(
        f'{run.entity}/{run.project}/{model_artifact_name}',
        type="model"
    )

    artifact_dir = artifact.download(train_config["save_dir"])
    ckpt_path = f'{artifact_dir}/{train_config["save_name"]}.pt'

    loaded_state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(
        loaded_state_dict
    )
else:
    assert logger_kwargs["id"] is None, \
        "When creating new WandB run, ID should be empty."

    run = wandb.init(
        config=config,
        project=logger_kwargs["project"],
        id=logger_kwargs["id"],
        resume="allow"
    )

    # update config with new run id
    config["autoencoder"]["train"]["resume"] = True
    config["autoencoder"]["train"]["logger_kwargs"]["id"] = run.id

    with open("./config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

optimizer = torch.optim.AdamW(model.model.parameters(), lr=learning_rate)
# discr_optimizer = torch.optim.AdamW(model.discriminator.parameters(), lr=1e-4)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=train_config["lr_warmup_steps"],
    num_training_steps=(len(train_dataloader) * num_epochs),
)

# wandb.watch(model, log_freq=20)


def train_loop(model, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    model, train_dataloader, lr_scheduler = accelerator.prepare(
        model, train_dataloader, lr_scheduler
    )

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                loss, mse_loss, emb_loss = model.training_step(batch)

                accelerator.backward(loss, retain_graph=True)
                # accelerator.backward(disct_loss)

                accelerator.clip_grad_norm_(model.parameters(), 4.0)

                optimizer.step()
                # discr_optimizer.step()

                lr_scheduler.step()

            optimizer.zero_grad()
            # discr_optimizer.zero_grad()

            accelerator.print(
                f"Step: {step}, loss: {loss}, emb_loss {emb_loss}, mse_loss {mse_loss}")

            if step % train_config["checkpoint_every"] == 0 and step != 0:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), save_path)

                artifact = wandb.Artifact(f"model-{run.id}", "model")
                artifact.add_file(save_path)
                wandb.log_artifact(artifact, aliases=["best", "latest"])

            if step % 200 == 0 and step != 0:
                with torch.no_grad():
                    images, labels = batch
                    reconstructed = model(images, inference=True)

                images = []
                for pianoroll in reconstructed:
                    image = Image.fromarray(np.uint8(pianoroll.T * 127))
                    images.append(
                        wandb.Image(image)
                    )

                wandb.log({"samples": images})

            if step % 20 == 0:
                wandb.log({
                    "loss": loss,
                    "emb_loss": emb_loss,
                    "mse_loss": mse_loss,
                })


train_loop(model, optimizer, train_dataloader, lr_scheduler)

if __name__ == "__main__":
    main()
