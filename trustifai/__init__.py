from trustifai.core import Trustifai
from trustifai.metrics import BaseMetric
from trustifai.structures import MetricContext, MetricResult
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except (LookupError, OSError):
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    raise RuntimeError(f"Failed to download 'punkt_tab':\n{e}")
