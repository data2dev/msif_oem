"""
Signal Generator
=================
Converts raw ESGD alpha predictions into actionable trading signals.
"""

import logging
import numpy as np
import config as cfg

log = logging.getLogger(__name__)


class SignalGenerator:
    """Converts alpha factors to trading signals."""

    def __init__(self):
        self.last_signals = {}  # symbol -> signal

    def generate(self, predictions: dict, confidences: dict) -> dict:
        """
        Generate trading signals from alpha predictions.
        
        Args:
            predictions: {symbol: alpha_value}
            confidences: {symbol: confidence_score}
            
        Returns:
            {symbol: {"signal": float, "direction": str, "strength": float, "confidence": float}}
        """
        signals = {}

        for symbol in predictions:
            alpha = float(predictions[symbol])
            conf = float(confidences.get(symbol, 1.0))

            # Adjust signal by confidence
            adjusted = alpha * conf

            # Direction
            if adjusted > cfg.MIN_SIGNAL_THRESHOLD:
                direction = "long"
            elif adjusted < -cfg.MIN_SIGNAL_THRESHOLD:
                direction = "short"
            else:
                direction = "flat"

            strength = abs(adjusted)

            signals[symbol] = {
                "signal": adjusted,
                "direction": direction,
                "strength": strength,
                "confidence": conf,
                "raw_alpha": alpha,
            }

        self.last_signals = signals
        return signals
