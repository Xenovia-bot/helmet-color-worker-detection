import pytest
import numpy as np
from unittest.mock import Mock, patch

from app.services.detector import ObjectDetector
from app.config.settings import Settings


class TestObjectDetector:
    """Test cases for the ObjectDetector class."""

    def test_init_with_valid_model(self):
        """Test initialization with a valid model."""
        with patch('app.services.detector.torch') as mock_torch:
            mock_model = Mock()
            mock_torch.hub.load.return_value = mock_model

            detector = ObjectDetector()

            assert detector.model is not None
            mock_torch.hub.load.assert_called()

    def test_init_fallback_to_default_model(self):
        """Test initialization falls back to default model when custom model fails."""
        with patch('app.services.detector.torch') as mock_torch:
            # Make first call raise an exception, second call succeed
            mock_torch.hub.load.side_effect = [Exception("Model not found"), Mock()]

            detector = ObjectDetector()

            assert detector.model is not None
            assert mock_torch.hub.load.call_count == 2

    @patch('app.services.detector.asyncio')
    async def test_detect_objects(self, mock_asyncio):
        """Test object detection functionality."""
        # Create a mock detector
        detector = ObjectDetector()
        detector.model = Mock()

        # Create a mock result
        mock_result = Mock()
        mock_df = Mock()
        mock_df.__getitem__.return_value = Mock()
        mock_df.__getitem__.return_value.__eq__.return_value = Mock()
        mock_result.pandas.return_value.xyxy = [mock_df]
        detector.model.return_value = mock_result

        # Create a dummy image
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Mock the event loop
        mock_loop = Mock()
        mock_asyncio.get_event_loop.return_value = mock_loop
        mock_loop.run_in_executor.return_value = mock_result

        # Call the method
        result = await detector.detect_objects(dummy_image)

        # Assertions
        assert isinstance(result, dict)
        assert "people_count" in result
        assert "violations" in result

    def test_process_results(self):
        """Test processing of detection results."""
        detector = ObjectDetector()

        # Create mock results that simulate pandas DataFrame
        mock_results = Mock()
        mock_df = Mock()

        # Mock the pandas DataFrame structure
        mock_person_row = Mock()
        mock_person_row.name = 'person'
        mock_person_row.xmin = 10.0
        mock_person_row.ymin = 20.0
        mock_person_row.xmax = 50.0
        mock_person_row.ymax = 80.0
        mock_person_row.confidence = 0.95

        mock_df.__iter__ = Mock(return_value=iter([mock_person_row]))
        mock_df.__getitem__ = Mock(return_value=Mock(__eq__=Mock(return_value=[mock_person_row])))
        mock_df.__len__ = Mock(return_value=1)

        mock_results.pandas.return_value.xyxy = [mock_df]

        # Process results
        result = detector._process_results(mock_results)

        assert isinstance(result, dict)
        assert "people_count" in result
        assert "violations" in result