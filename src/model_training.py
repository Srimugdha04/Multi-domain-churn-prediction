
"""Train and compare telecom-to-bank churn prediction models."""

import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from tensorflow.keras.optimizers import Adam

from src.data_preprocessing import load_and_preprocess_data


SEED = 42
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

EPOCHS = 100
BATCH_SIZE = 32


def set_seeds():
    """Set random seeds for reproducibility."""
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


def get_class_weights(y):
    """Calculate class weights using training labels only."""
    classes = np.unique(y)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y.astype(int),
    )

    return {
        int(label): float(weight)
        for label, weight in zip(classes, weights)
    }


def get_callbacks(checkpoint_path):
    """Create validation-based training callbacks."""
    return [
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def compile_model(model, learning_rate):
    """Compile a binary churn classifier."""
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            "accuracy",
        ],
    )


def build_classifier(input_dim, name):
    """Build the telecom or bank baseline classifier."""
    inputs = Input(shape=(input_dim,), name=f"{name}_input")

    x = Dense(
        32, activation="relu", name=f"{name}_dense_1"
    )(inputs)
    x = Dropout(0.2)(x)

    x = Dense(
        16, activation="relu", name=f"{name}_dense_2"
    )(x)
    x = Dropout(0.2)(x)

    outputs = Dense(
        1, activation="sigmoid", name=f"{name}_output"
    )(x)

    return Model(inputs, outputs, name=name)


def train_model(model, data, checkpoint, learning_rate=0.001):
    """Train a model and reload its best validation checkpoint."""
    compile_model(model, learning_rate)

    history = model.fit(
        data["X_train"],
        data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=get_class_weights(data["y_train"]),
        callbacks=get_callbacks(checkpoint),
        verbose=1,
    )

    best_model = tf.keras.models.load_model(
        checkpoint, compile=False
    )

    return best_model, history


def train_telecom_source(data):
    """Train the source encoder on telecom data."""
    print("\nTraining telecom source model...")

    set_seeds()

    model = build_classifier(
        input_dim=data["X_train"].shape[1],
        name="telecom_source",
    )

    checkpoint = ARTIFACT_DIR / "telecom_source_best.keras"

    return train_model(model, data, checkpoint)


def train_bank_baseline(data):
    """Train a bank-only model without transferred weights."""
    print("\nTraining bank-only baseline...")

    set_seeds()

    model = build_classifier(
        input_dim=data["X_train"].shape[1],
        name="bank_baseline",
    )

    checkpoint = ARTIFACT_DIR / "bank_baseline_best.keras"

    return train_model(model, data, checkpoint)


def build_transfer_model(
    source_model,
    bank_input_dim,
    use_pretrained=True,
):
    """Build a bank adapter and a source-shaped encoder."""

    bank_inputs = Input(
        shape=(bank_input_dim,),
        name="bank_features",
    )

    source_dense_1 = source_model.get_layer(
        "telecom_source_dense_1"
    )
    source_dense_2 = source_model.get_layer(
        "telecom_source_dense_2"
    )

    # Project bank features into the telecom encoder input space.
    adapted = Dense(
        source_model.input_shape[-1],
        activation="linear",
        name="bank_adapter",
    )(bank_inputs)

    # Create independent encoder layers with matching architecture.
    x = Dense(
        source_dense_1.units,
        activation=source_dense_1.activation,
        name="transfer_dense_1",
    )(adapted)

    x = Dropout(0.2, name="transfer_dropout_1")(x)

    x = Dense(
        source_dense_2.units,
        activation=source_dense_2.activation,
        name="transfer_dense_2",
    )(x)

    x = Dropout(0.2, name="transfer_dropout_2")(x)

    outputs = Dense(
        1,
        activation="sigmoid",
        name="bank_transfer_output",
    )(x)

    model = Model(
        inputs=bank_inputs,
        outputs=outputs,
        name="telecom_to_bank_transfer",
    )

    encoder_1 = model.get_layer("transfer_dense_1")
    encoder_2 = model.get_layer("transfer_dense_2")

    if use_pretrained:
        # Copy learned telecom encoder weights.
        encoder_1.set_weights(source_dense_1.get_weights())
        encoder_2.set_weights(source_dense_2.get_weights())

    # Stage 1: freeze the encoder.
    encoder_1.trainable = False
    encoder_2.trainable = False

    return model


def train_transfer_model(
    source_model,
    data,
    use_pretrained=True,
):
    """Train the transfer model in two stages."""

    experiment = (
        "pretrained_transfer"
        if use_pretrained
        else "random_transfer_control"
    )

    print(f"\nTraining {experiment}...")

    # Reset seeds so both experiments start with matching
    # adapter and prediction-head initialization.
    set_seeds()

    model = build_transfer_model(
        source_model,
        data["X_train"].shape[1],
        use_pretrained=use_pretrained,
    )

    stage_1_checkpoint = (
        ARTIFACT_DIR / f"{experiment}_stage1_best.keras"
    )

    print("\nStage 1: Train adapter and prediction head.")

    model, history_stage_1 = train_model(
        model,
        data,
        stage_1_checkpoint,
        learning_rate=0.001,
    )

    # Restore the best stage-one model before fine-tuning.
    encoder_1 = model.get_layer("transfer_dense_1")
    encoder_2 = model.get_layer("transfer_dense_2")

    encoder_1.trainable = True
    encoder_2.trainable = True

    print("\nStage 2: Fine-tune the encoder.")

    stage_2_checkpoint = (
        ARTIFACT_DIR / f"{experiment}_finetuned_best.keras"
    )

    model, history_stage_2 = train_model(
        model,
        data,
        stage_2_checkpoint,
        learning_rate=0.0001,
    )

    return model, history_stage_1, history_stage_2


def main():
    set_seeds()

    print("Loading and preprocessing datasets...")

    datasets = load_and_preprocess_data()

    telecom = datasets["telecom"]
    bank = datasets["bank"]

    # Experiment 1: source-domain training.
    source_model, source_history = train_telecom_source(
        telecom
    )

    # Experiment 2: target-domain baseline.
    baseline_model, baseline_history = train_bank_baseline(
        bank
    )

    # Experiment 3: pretrained transfer learning.
    transfer_model, transfer_history_1, transfer_history_2 = (
        train_transfer_model(
            source_model,
            bank,
            use_pretrained=True,
        )
    )

    # Experiment 4: same architecture, random encoder initialization.
    random_model, random_history_1, random_history_2 = (
        train_transfer_model(
            source_model,
            bank,
            use_pretrained=False,
        )
    )

    print("\nAll four training experiments completed.")

    print("\nGenerated model artifacts:")
    for path in sorted(ARTIFACT_DIR.glob("*.keras")):
        print(path)


if __name__ == "__main__":
    main()
