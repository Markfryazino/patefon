from email.policy import default
import click

from patefon import train_patefon, create_dataset_of_diagrams, set_random_seed


@click.command()
@click.option("--objects-per-class", default=1000, help="How many point clouds to generate per each of 5 classes")
@click.option("--n-encoders", default=5, help="Number of encoder blocks in Patefon")
@click.option("--embedding-dim", default=128, help="Dimension of homologies embedding")
@click.option("--decoder-dropout", default=0.0, help="Dropout rate in decoder blocks")
@click.option("--batch-size", default=64, help="Batch size")
@click.option("--epochs", default=500, help="Number of epochs to train")
@click.option("--random-state", default=42, help="Random seed used for training and data generation")
@click.option("--wandb-project", default=None, help="wandb project name")
@click.option("--wandb-entity", default=None, help="wandb entity name")
@click.option("--wandb-run", default=None, help="wandb run name")
def run(
    objects_per_class: int,
    n_encoders: int,
    embedding_dim: int,
    decoder_dropout: float,
    batch_size: int,
    epochs: int,
    random_state: int,
    wandb_project: str,
    wandb_entity: str,
    wandb_run: str,
):
    updated_params = dict(
        batch_size=batch_size,
        epochs=epochs,
        random_state=random_state,
        wandb=dict(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run,
        ),
        patefon=dict(
            n_encoders=n_encoders,
            embedding_dim=embedding_dim,
            decoder_kwargs=dict(
                dropout_prob=decoder_dropout,
            ),
        ),
    )

    set_random_seed(random_state)

    D_train, D_test, y_train, y_test = create_dataset_of_diagrams(objects_per_class)
    train_patefon(D_train, y_train, D_test, y_test, params=updated_params)


if __name__ == "__main__":
    run()
