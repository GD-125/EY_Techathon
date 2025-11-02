"""
Notification Service for sending alerts and updates
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Types of notifications"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class NotificationService:
    """
    Service for managing and sending notifications across multiple channels
    """

    def __init__(self):
        self.notification_queue = []
        self.sent_notifications = []

    async def send_notification(
        self,
        recipient: str,
        subject: str,
        message: str,
        notification_type: NotificationType = NotificationType.EMAIL,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Send a notification to a recipient

        Args:
            recipient: Recipient identifier (email, phone, user_id)
            subject: Notification subject
            message: Notification message
            notification_type: Type of notification
            priority: Priority level
            metadata: Additional metadata

        Returns:
            Dictionary with send status
        """
        try:
            notification = {
                'id': f"notif_{datetime.now().timestamp()}",
                'recipient': recipient,
                'subject': subject,
                'message': message,
                'type': notification_type.value,
                'priority': priority.value,
                'timestamp': datetime.now().isoformat(),
                'status': 'sent',
                'metadata': metadata or {}
            }

            # In production, integrate with actual notification services
            # (SendGrid, Twilio, Firebase, etc.)
            logger.info(f"Sending {notification_type.value} notification to {recipient}")

            self.sent_notifications.append(notification)

            return {
                'success': True,
                'notification_id': notification['id'],
                'message': f'Notification sent successfully via {notification_type.value}'
            }

        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def send_loan_application_notification(
        self,
        applicant_email: str,
        application_id: str,
        status: str
    ) -> Dict:
        """Send notification about loan application status"""

        status_messages = {
            'submitted': {
                'subject': 'Loan Application Received',
                'message': f'Your loan application (ID: {application_id}) has been received and is under review. We will notify you once the review is complete.'
            },
            'approved': {
                'subject': 'Loan Application Approved',
                'message': f'Congratulations! Your loan application (ID: {application_id}) has been approved. Please check your dashboard for next steps.'
            },
            'rejected': {
                'subject': 'Loan Application Status',
                'message': f'Unfortunately, your loan application (ID: {application_id}) could not be approved at this time. Please review the reasons in your dashboard.'
            },
            'more_info_required': {
                'subject': 'Additional Information Required',
                'message': f'We need additional information for your loan application (ID: {application_id}). Please check your dashboard for details.'
            }
        }

        msg_data = status_messages.get(status, {
            'subject': 'Loan Application Update',
            'message': f'Your loan application (ID: {application_id}) status has been updated.'
        })

        return await self.send_notification(
            recipient=applicant_email,
            subject=msg_data['subject'],
            message=msg_data['message'],
            notification_type=NotificationType.EMAIL,
            priority=NotificationPriority.HIGH if status in ['approved', 'rejected'] else NotificationPriority.MEDIUM,
            metadata={'application_id': application_id, 'status': status}
        )

    async def send_document_verification_notification(
        self,
        applicant_email: str,
        application_id: str,
        document_type: str,
        verification_status: str
    ) -> Dict:
        """Send notification about document verification"""

        if verification_status == 'verified':
            message = f'Your {document_type} for application {application_id} has been successfully verified.'
        elif verification_status == 'rejected':
            message = f'Your {document_type} for application {application_id} could not be verified. Please upload a clearer document.'
        else:
            message = f'Your {document_type} for application {application_id} is under verification.'

        return await self.send_notification(
            recipient=applicant_email,
            subject=f'Document Verification: {document_type}',
            message=message,
            notification_type=NotificationType.EMAIL,
            priority=NotificationPriority.MEDIUM,
            metadata={
                'application_id': application_id,
                'document_type': document_type,
                'verification_status': verification_status
            }
        )

    async def send_bulk_notifications(
        self,
        recipients: List[str],
        subject: str,
        message: str,
        notification_type: NotificationType = NotificationType.EMAIL
    ) -> Dict:
        """Send bulk notifications to multiple recipients"""

        results = []
        for recipient in recipients:
            result = await self.send_notification(
                recipient=recipient,
                subject=subject,
                message=message,
                notification_type=notification_type
            )
            results.append(result)

        success_count = sum(1 for r in results if r.get('success'))

        return {
            'total': len(recipients),
            'success': success_count,
            'failed': len(recipients) - success_count,
            'results': results
        }

    async def send_agent_alert(
        self,
        agent_type: str,
        alert_message: str,
        priority: NotificationPriority = NotificationPriority.HIGH
    ) -> Dict:
        """Send alert to system agents"""

        return await self.send_notification(
            recipient=f"agent_{agent_type}",
            subject=f"Agent Alert: {agent_type}",
            message=alert_message,
            notification_type=NotificationType.IN_APP,
            priority=priority,
            metadata={'agent_type': agent_type}
        )

    def get_notification_history(
        self,
        recipient: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Get notification history"""

        if recipient:
            notifications = [
                n for n in self.sent_notifications
                if n['recipient'] == recipient
            ]
        else:
            notifications = self.sent_notifications

        # Return most recent notifications
        return sorted(
            notifications,
            key=lambda x: x['timestamp'],
            reverse=True
        )[:limit]

    def get_notification_stats(self) -> Dict:
        """Get notification statistics"""

        total = len(self.sent_notifications)
        by_type = {}
        by_priority = {}

        for notif in self.sent_notifications:
            notif_type = notif['type']
            priority = notif['priority']

            by_type[notif_type] = by_type.get(notif_type, 0) + 1
            by_priority[priority] = by_priority.get(priority, 0) + 1

        return {
            'total_sent': total,
            'by_type': by_type,
            'by_priority': by_priority
        }
