"""Safety equipment analyzer service."""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config.settings import Settings
from app.core.constants import EquipmentType
from app.core.exceptions import ImageProcessingError
from app.domain.detector_base import BaseDetector


logger = logging.getLogger(__name__)


@dataclass
class PersonDetection:
    """Represents a detected person with their safety equipment status."""
    person_id: int
    bbox: list[float]
    confidence: float
    equipment: dict[str, bool] = field(default_factory=dict)
    missing_equipment: list[str] = field(default_factory=list)

    def check_safety_compliance(self, required_equipment: list[EquipmentType]) -> None:
        """Check if person has all required safety equipment.

        Args:
            required_equipment: List of required equipment types.
        """
        self.missing_equipment = [
            eq.value for eq in required_equipment
            if not self.equipment.get(eq.value, False)
        ]

    @property
    def is_compliant(self) -> bool:
        """Check if person is compliant (has all equipment)."""
        return len(self.missing_equipment) == 0


@dataclass
class SafetyViolation:
    """Represents a safety violation for a person."""
    person_id: int
    missing_equipment: list[str]
    bbox: list[float]
    confidence: float


@dataclass
class SafetyAnalysisResult:
    """Result of safety equipment analysis."""
    people_count: int
    violations: list[SafetyViolation]
    total_people_analyzed: int = 0

    @property
    def has_violations(self) -> bool:
        """Check if any violations were found."""
        return len(self.violations) > 0

    @property
    def compliance_rate(self) -> float:
        """Calculate the compliance rate."""
        if self.total_people_analyzed == 0:
            return 0.0
        compliant = self.total_people_analyzed - len(self.violations)
        return compliant / self.total_people_analyzed


class SafetyAnalyzer:
    """Analyzes detections to identify safety equipment violations."""

    def __init__(
        self,
        required_equipment: list[EquipmentType] | None = None,
        proximity_threshold: float = 0.6
    ):
        """Initialize the safety analyzer.

        Args:
            required_equipment: List of required safety equipment.
            proximity_threshold: Multiplier for person height to determine detection range.
        """
        self._settings = Settings()
        self._required_equipment = required_equipment or [
            EquipmentType.HELMET,
            EquipmentType.VEST
        ]
        self._proximity_threshold = proximity_threshold

    def analyze(
        self,
        detections: list[dict[str, Any]],
        image_shape: tuple[int, int, int] | None = None
    ) -> SafetyAnalysisResult:
        """Analyze detections to find safety violations.

        Args:
            detections: List of detection dictionaries from the detector.
            image_shape: Optional image shape (h, w, c) for spatial checks.

        Returns:
            SafetyAnalysisResult with violation details.
        """
        if not detections:
            return SafetyAnalysisResult(people_count=0, violations=[])

        people = self._extract_people(detections)
        violations = []

        for idx, person in enumerate(people):
            person_det = self._analyze_person_equipment(person, detections, idx)
            person_det.check_safety_compliance(self._required_equipment)

            if not person_det.is_compliant:
                violations.append(SafetyViolation(
                    person_id=person_det.person_id,
                    missing_equipment=person_det.missing_equipment,
                    bbox=person_det.bbox,
                    confidence=person_det.confidence
                ))

        return SafetyAnalysisResult(
            people_count=len(people),
            violations=violations,
            total_people_analyzed=len(people)
        )

    def _extract_people(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract person detections from all detections.

        Args:
            detections: All detection results.

        Returns:
            List of person detection dictionaries.
        """
        min_conf = self._settings.MIN_PERSON_ANALYSIS_CONFIDENCE
        people = []
        for d in detections:
            class_name = d.get("class_name", d.get("class", "")).lower()
            class_id = d.get("class", 0)

            if class_name == "person" or class_id == 0:
                if d.get("confidence", 0) >= min_conf:
                    people.append(d)

        return people

    def _analyze_person_equipment(
        self,
        person: dict[str, Any],
        all_detections: list[dict[str, Any]],
        person_idx: int
    ) -> PersonDetection:
        """Analyze what equipment a person has.

        Args:
            person: Person detection dictionary.
            all_detections: All detection results.
            person_idx: Index of this person.

        Returns:
            PersonDetection with equipment status.
        """
        person_bbox = person["bbox"]
        equipment: dict[str, bool] = {}

        for equipment_type in self._required_equipment:
            equipment[equipment_type.value] = self._check_equipment_near_person(
                equipment_type.value,
                person_bbox,
                all_detections
            )

        return PersonDetection(
            person_id=person_idx + 1,
            bbox=[float(x) for x in person_bbox],
            confidence=person["confidence"],
            equipment=equipment
        )

    def _check_equipment_near_person(
        self,
        equipment_class: str,
        person_bbox: list[float],
        all_detections: list[dict[str, Any]]
    ) -> bool:
        """Check if equipment of given class is near the person.

        Args:
            equipment_class: Name of equipment class to check.
            person_bbox: Person's bounding box [xmin, ymin, xmax, ymax].
            all_detections: All detection results.

        Returns:
            True if equipment is found near person, False otherwise.
        """
        # bbox format: [cx, cy, w, h] normalized 0-1
        person_x_center = person_bbox[0]
        person_y_center = person_bbox[1]
        person_height = person_bbox[3]

        proximity_threshold = person_height * self._proximity_threshold

        # Map general equipment types to model class variations
        helmet_classes = {"helmet", "blue helmet", "red helmet", "white helmet", "yellow helmet"}
        vest_classes = {"vest"}

        target_classes: set[str]
        if equipment_class.lower() == "helmet":
            target_classes = helmet_classes
        elif equipment_class.lower() == "vest":
            target_classes = vest_classes
        else:
            target_classes = {equipment_class.lower()}

        for detection in all_detections:
            class_name = detection.get("class_name", detection.get("class", "")).lower()

            if class_name not in target_classes:
                continue

            equip_bbox = detection["bbox"]
            equip_x_center = equip_bbox[0]
            equip_y_center = equip_bbox[1]

            distance = np.sqrt(
                (equip_x_center - person_x_center) ** 2 +
                (equip_y_center - person_y_center) ** 2
            )

            if distance < proximity_threshold:
                return True

        return False

    @property
    def required_equipment(self) -> list[EquipmentType]:
        """Get list of required equipment."""
        return self._required_equipment.copy()

    def set_required_equipment(self, equipment: list[EquipmentType]) -> None:
        """Set required equipment list.

        Args:
            equipment: New list of required equipment.
        """
        self._required_equipment = equipment