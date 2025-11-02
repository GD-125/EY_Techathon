"""
Behavioral Analysis Service
Analyzes conversation patterns for personality detection and trust scoring
"""

from typing import List
from ..models.schemas import PersonalityType


class BehavioralAnalyzer:
    """
    Analyzes customer conversation patterns
    Creates behavioral trust scores and personality profiles
    """

    def __init__(self):
        self.response_times = []
        self.message_lengths = []
        self.question_count = 0
        self.total_messages = 0
        self.hesitation_markers = 0
        self.confidence_markers = 0

    def analyze_message(self, message: str, response_time: float = 2.0):
        """
        Analyze a single customer message

        Args:
            message: Customer message
            response_time: Response time in seconds
        """
        self.total_messages += 1
        self.response_times.append(response_time)
        self.message_lengths.append(len(message.split()))

        # Detect questions
        if '?' in message:
            self.question_count += 1

        # Detect hesitation markers
        hesitation_words = [
            'maybe', 'perhaps', 'not sure', 'i think', 'probably',
            'might', 'possibly', 'uncertain', 'confused'
        ]
        if any(word in message.lower() for word in hesitation_words):
            self.hesitation_markers += 1

        # Detect confidence markers
        confidence_words = [
            'definitely', 'sure', 'certain', 'absolutely', 'yes',
            'confirm', 'positive', 'confident', 'know'
        ]
        if any(word in message.lower() for word in confidence_words):
            self.confidence_markers += 1

    def get_personality_type(self) -> PersonalityType:
        """
        Determine customer personality type

        Returns:
            PersonalityType enum
        """
        if self.total_messages < 2:
            return PersonalityType.AMIABLE  # Default

        avg_response_time = sum(self.response_times) / len(self.response_times)
        avg_message_length = sum(self.message_lengths) / len(self.message_lengths)
        question_ratio = self.question_count / self.total_messages if self.total_messages > 0 else 0

        # Analytical: Many questions, longer messages, slower responses
        if question_ratio > 0.3 and avg_message_length > 15:
            return PersonalityType.ANALYTICAL

        # Driver: Quick responses, short messages, confident
        elif avg_response_time < 3 and avg_message_length < 10:
            return PersonalityType.DRIVER

        # Expressive: Quick responses, enthusiastic, longer messages
        elif avg_response_time < 4 and self.confidence_markers > self.hesitation_markers:
            return PersonalityType.EXPRESSIVE

        # Amiable: Balanced, moderate pace
        else:
            return PersonalityType.AMIABLE

    def get_behavioral_trust_score(self) -> float:
        """
        Calculate behavioral trust score (0-100)

        Returns:
            Trust score
        """
        if self.total_messages < 2:
            return 70.0  # Neutral starting score

        score = 70.0

        # Consistent response times indicate genuine engagement
        if len(self.response_times) > 2:
            avg_time = sum(self.response_times) / len(self.response_times)
            time_variance = sum(
                (t - avg_time) ** 2 for t in self.response_times
            ) / len(self.response_times)

            if time_variance < 2.0:  # Low variance is good
                score += 10

        # Confidence over hesitation
        if self.confidence_markers > self.hesitation_markers:
            score += 10
        elif self.hesitation_markers > self.confidence_markers * 2:
            score -= 15

        # Reasonable message lengths
        if len(self.message_lengths) > 0:
            avg_length = sum(self.message_lengths) / len(self.message_lengths)
            if 5 < avg_length < 30:
                score += 10

        # Question asking shows genuine interest
        if self.question_count > 0:
            score += 5

        return max(0, min(100, score))

    def get_risk_flags(self) -> List[str]:
        """
        Identify potential risk indicators

        Returns:
            List of risk flags
        """
        flags = []

        if self.total_messages == 0:
            return flags

        if self.hesitation_markers > self.total_messages * 0.5:
            flags.append("High hesitation in responses")

        if len(self.response_times) > 2:
            avg_time = sum(self.response_times) / len(self.response_times)
            if avg_time > 15:
                flags.append("Unusually long response times")
            elif avg_time < 1:
                flags.append("Suspiciously fast responses")

        if self.question_count == 0 and self.total_messages > 5:
            flags.append("No clarification questions asked")

        # Check for copy-paste patterns (very long messages)
        if len(self.message_lengths) > 0:
            max_length = max(self.message_lengths)
            avg_length = sum(self.message_lengths) / len(self.message_lengths)
            if max_length > avg_length * 3 and max_length > 50:
                flags.append("Possible copy-paste responses detected")

        return flags

    def get_metrics(self) -> dict:
        """Get all behavioral metrics"""
        return {
            "total_messages": self.total_messages,
            "avg_response_time": sum(self.response_times) / len(self.response_times) if self.response_times else 0,
            "avg_message_length": sum(self.message_lengths) / len(self.message_lengths) if self.message_lengths else 0,
            "question_ratio": self.question_count / self.total_messages if self.total_messages > 0 else 0,
            "hesitation_markers": self.hesitation_markers,
            "confidence_markers": self.confidence_markers,
            "personality_type": self.get_personality_type().value,
            "trust_score": self.get_behavioral_trust_score(),
            "risk_flags": self.get_risk_flags()
        }

    def reset(self):
        """Reset analyzer for new session"""
        self.__init__()
