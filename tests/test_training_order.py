import types
import numpy as np
import tensorflow as tf
from stock_direction_trader import StockDirectionTrader


def test_training_no_shuffle_and_chronological_split(monkeypatch):
    trader = StockDirectionTrader("TEST", lookback_days=1)

    # small dummy dataset
    X = np.random.rand(30, 1, 1)
    y = np.random.randint(0, 2, size=30)

    captured = {}
    dummy_history = types.SimpleNamespace(history={"val_loss": [0.0]})

    def fake_fit(self, *args, **kwargs):
        captured["shuffle"] = kwargs.get("shuffle")
        return dummy_history

    monkeypatch.setattr(tf.keras.Model, "fit", fake_fit)

    trader.train_model(X, y, epochs=1, batch_size=4)

    assert captured["shuffle"] is False
    assert max(trader.last_train_indices) < min(trader.last_val_indices)
